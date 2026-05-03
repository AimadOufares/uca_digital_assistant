const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const typingIndicator = document.getElementById("typingIndicator");
const promptButtons = document.querySelectorAll(".prompt-btn");
const conversationList = document.getElementById("conversationList");
const newConversationButton = document.getElementById("newConversationButton");
const conversationTitle = document.getElementById("conversationTitle");

const API_URL = "/api/chat/";
const CONVERSATIONS_API_URL = "/api/chat/conversations/";
let historyLoaded = false;
let currentConversationId = null;

function getCookie(name) {
    const cookieValue = document.cookie
        .split(";")
        .map((item) => item.trim())
        .find((item) => item.startsWith(`${name}=`));
    return cookieValue ? decodeURIComponent(cookieValue.split("=").slice(1).join("=")) : "";
}

function getCsrfToken() {
    const csrfTokenInput = document.querySelector("[name=csrfmiddlewaretoken]");
    return (csrfTokenInput && csrfTokenInput.value) || getCookie("csrftoken") || "";
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function scrollToBottom() {
    const scrollArea = document.querySelector(".chat-scroll-area") || chatMessages;
    scrollArea.scrollTop = scrollArea.scrollHeight;
}

function autoResizeInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 140)}px`;
}

function clearInitialMessages() {
    chatMessages.innerHTML = "";
}

function setConversationTitle(title) {
    if (!conversationTitle) {
        return;
    }
    conversationTitle.textContent = title || "Nouvelle conversation";
}

function buildConversationPreview(item) {
    const preview = (item.preview || "").trim();
    if (!preview) {
        return "Conversation prete a commencer";
    }
    return preview.length > 74 ? `${preview.slice(0, 74)}...` : preview;
}

function renderConversationList(conversations = []) {
    if (!conversationList) {
        return;
    }
    if (!Array.isArray(conversations) || !conversations.length) {
        conversationList.innerHTML = `<div class="conversation-empty">Commencez une conversation pour voir votre historique ici.</div>`;
        return;
    }
    conversationList.innerHTML = conversations
        .map((item) => {
            const classes = item.selected ? "conversation-item is-active" : "conversation-item";
            const countLabel = `${item.message_count || 0} message${item.message_count === 1 ? "" : "s"}`;
            return `
                <div class="${classes}" data-conversation-id="${item.id}">
                    <button type="button" class="conversation-main" data-open-conversation="${item.id}">
                        <span class="conversation-item-title">${escapeHtml(item.title || "Nouvelle conversation")}</span>
                        <span class="conversation-item-preview">${escapeHtml(buildConversationPreview(item))}</span>
                        <span class="conversation-item-meta">${escapeHtml(countLabel)}</span>
                    </button>
                    <div class="conversation-actions">
                        <button type="button" class="conversation-action-btn" data-rename-conversation="${item.id}">Renommer</button>
                        <button type="button" class="conversation-action-btn conversation-action-danger" data-archive-conversation="${item.id}">Archiver</button>
                    </div>
                </div>
            `;
        })
        .join("");

    conversationList.querySelectorAll("[data-open-conversation]").forEach((button) => {
        button.addEventListener("click", async () => {
            const conversationId = Number(button.dataset.openConversation || "0");
            if (!conversationId || conversationId === currentConversationId || sendButton.disabled) {
                return;
            }
            await loadConversationHistory(conversationId);
        });
    });

    conversationList.querySelectorAll("[data-rename-conversation]").forEach((button) => {
        button.addEventListener("click", async (event) => {
            event.stopPropagation();
            const conversationId = Number(button.dataset.renameConversation || "0");
            if (!conversationId) {
                return;
            }
            const currentTitle = button.closest("[data-conversation-id]")?.querySelector(".conversation-item-title")?.textContent || "";
            const nextTitle = window.prompt("Nouveau titre de la conversation", currentTitle.trim());
            if (!nextTitle || !nextTitle.trim()) {
                return;
            }
            await updateConversation(conversationId, { title: nextTitle.trim() });
        });
    });

    conversationList.querySelectorAll("[data-archive-conversation]").forEach((button) => {
        button.addEventListener("click", async (event) => {
            event.stopPropagation();
            const conversationId = Number(button.dataset.archiveConversation || "0");
            if (!conversationId) {
                return;
            }
            const confirmed = window.confirm("Archiver cette conversation de votre historique actif ?");
            if (!confirmed) {
                return;
            }
            await archiveConversation(conversationId);
        });
    });
}

function buildSourceMarkup(sources) {
    if (!Array.isArray(sources) || !sources.length) {
        return "";
    }
    const items = sources
        .slice(0, 3)
        .map((source) => {
            const name = escapeHtml(source.name || source.path || "Source");
            const score = typeof source.score === "number" ? ` <span class="source-score">${source.score.toFixed(2)}</span>` : "";
            return `<li>${name}${score}</li>`;
        })
        .join("");
    return `<div class="message-sources"><p>Sources utiles</p><ul>${items}</ul></div>`;
}

function buildMetaMarkup(confidence) {
    if (!confidence) {
        return "";
    }
    return `<div class="message-meta">Confiance: ${escapeHtml(confidence)}</div>`;
}

function appendMessage(role, text, options = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "message-user" : "message-assistant"}`;

    if (role !== "user") {
        const avatar = document.createElement("div");
        avatar.className = "avatar avatar-ai";
        avatar.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>`;
        wrapper.appendChild(avatar);
    }

    const content = document.createElement("div");
    content.className = "message-content";

    const paragraph = document.createElement("p");
    paragraph.innerHTML = escapeHtml(text).replaceAll("\n", "<br>");
    content.appendChild(paragraph);

    if (role !== "user") {
        const metaMarkup = buildMetaMarkup(options.confidence || "");
        const sourceMarkup = buildSourceMarkup(options.sources || []);
        if (metaMarkup) {
            const meta = document.createElement("div");
            meta.innerHTML = metaMarkup;
            content.appendChild(meta.firstChild);
        }
        if (sourceMarkup) {
            const sources = document.createElement("div");
            sources.innerHTML = sourceMarkup;
            content.appendChild(sources.firstChild);
        }
    }

    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function setLoadingState(isLoading) {
    sendButton.disabled = isLoading;
    messageInput.disabled = isLoading;
    typingIndicator.hidden = !isLoading;
    if (newConversationButton) {
        newConversationButton.disabled = isLoading;
    }
    promptButtons.forEach((button) => {
        button.disabled = isLoading;
    });
    if (!isLoading) {
        messageInput.focus();
    }
}

async function submitMessage(message) {
    appendMessage("user", message);
    messageInput.value = "";
    autoResizeInput();
    setLoadingState(true);

    try {
        const csrfToken = getCsrfToken();
        const response = await fetch(API_URL, {
            method: "POST",
            credentials: "same-origin",
            headers: {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": csrfToken,
            },
            body: JSON.stringify({ message, conversation_id: currentConversationId }),
        });

        let payload = {};
        try {
            payload = await response.json();
        } catch (error) {
            payload = {};
        }

        if (!response.ok) {
            if (response.status === 403 || response.status === 401) {
                window.location.href = `/login/?next=${encodeURIComponent("/chat/")}`;
                return;
            }
            const detail = payload.detail || "Le serveur a retourne une erreur.";
            throw new Error(detail);
        }

        const answer = payload.answer || "Aucune reponse generee.";
        currentConversationId = payload.conversation_id || currentConversationId;
        setConversationTitle(payload.conversation_title || "Nouvelle conversation");
        renderConversationList(payload.conversations || []);
        appendMessage("assistant", answer, {
            confidence: payload.confidence || "",
            sources: payload.sources || [],
        });
    } catch (error) {
        appendMessage(
            "assistant",
            `Je n'ai pas pu traiter votre demande pour le moment.\nDetail: ${error.message || "Erreur reseau."}`
        );
    } finally {
        setLoadingState(false);
    }
}

async function loadConversationHistory(conversationId = null) {
    try {
        if (conversationId !== null) {
            setLoadingState(true);
        }
        const url = conversationId ? `${API_URL}?conversation_id=${encodeURIComponent(conversationId)}` : API_URL;
        const effectiveResponse = await fetch(url, {
            method: "GET",
            credentials: "same-origin",
            headers: {
                "X-Requested-With": "XMLHttpRequest",
            },
        });

        if (effectiveResponse.status === 403 || effectiveResponse.status === 401) {
            window.location.href = `/login/?next=${encodeURIComponent("/chat/")}`;
            return;
        }
        if (!effectiveResponse.ok) {
            return;
        }

        const payload = await effectiveResponse.json();
        currentConversationId = payload.conversation_id || null;
        setConversationTitle(payload.conversation_title || "Nouvelle conversation");
        renderConversationList(payload.conversations || []);
        const messages = Array.isArray(payload.messages) ? payload.messages : [];
        clearInitialMessages();
        if (!messages.length) {
            appendMessage("assistant", "Bonjour, je suis votre assistant UCA. Posez votre question pour commencer cette conversation.");
        } else {
            messages.forEach((message) => {
                appendMessage(message.role, message.content || "", {
                    confidence: message.confidence || "",
                    sources: message.sources || [],
                });
            });
        }
        historyLoaded = true;
    } catch (error) {
        historyLoaded = true;
    } finally {
        setLoadingState(false);
    }
}

async function createNewConversation() {
    if (!newConversationButton || sendButton.disabled) {
        return;
    }
    setLoadingState(true);
    try {
        const response = await fetch(CONVERSATIONS_API_URL, {
            method: "POST",
            credentials: "same-origin",
            headers: {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": getCsrfToken(),
            },
            body: JSON.stringify({}),
        });
        if (!response.ok) {
            throw new Error("Creation impossible pour le moment.");
        }
        const payload = await response.json();
        currentConversationId = payload.conversation_id || null;
        setConversationTitle(payload.conversation_title || "Nouvelle conversation");
        renderConversationList(payload.conversations || []);
        clearInitialMessages();
        appendMessage("assistant", "Nouvelle conversation ouverte. Decrivez votre besoin UCA et je m'appuierai sur les sources disponibles.");
    } catch (error) {
        appendMessage("assistant", `Je n'ai pas pu ouvrir une nouvelle conversation.\nDetail: ${error.message || "Erreur reseau."}`);
    } finally {
        setLoadingState(false);
    }
}

async function updateConversation(conversationId, payload) {
    setLoadingState(true);
    try {
        const response = await fetch(`/api/chat/conversations/${conversationId}/`, {
            method: "PATCH",
            credentials: "same-origin",
            headers: {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": getCsrfToken(),
            },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            throw new Error("Mise a jour impossible.");
        }
        const data = await response.json();
        renderConversationList(data.conversations || []);
        if (currentConversationId === conversationId && data.conversation) {
            setConversationTitle(data.conversation.title || "Nouvelle conversation");
        }
    } catch (error) {
        appendMessage("assistant", `Je n'ai pas pu mettre a jour cette conversation.\nDetail: ${error.message || "Erreur reseau."}`);
    } finally {
        setLoadingState(false);
    }
}

async function archiveConversation(conversationId) {
    let shouldOpenFreshConversation = false;
    setLoadingState(true);
    try {
        const response = await fetch(`/api/chat/conversations/${conversationId}/`, {
            method: "DELETE",
            credentials: "same-origin",
            headers: {
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": getCsrfToken(),
            },
        });
        if (!response.ok) {
            throw new Error("Archivage impossible.");
        }
        const data = await response.json();
        renderConversationList(data.conversations || []);
        if (currentConversationId === conversationId) {
            currentConversationId = null;
            shouldOpenFreshConversation = true;
        }
    } catch (error) {
        appendMessage("assistant", `Je n'ai pas pu archiver cette conversation.\nDetail: ${error.message || "Erreur reseau."}`);
    } finally {
        setLoadingState(false);
    }
    if (shouldOpenFreshConversation) {
        await createNewConversation();
    }
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) {
        return;
    }
    await submitMessage(message);
});

promptButtons.forEach((button) => {
    button.addEventListener("click", async () => {
        const prompt = (button.dataset.prompt || "").trim();
        if (!prompt || sendButton.disabled) {
            return;
        }
        messageInput.value = prompt;
        autoResizeInput();
        await submitMessage(prompt);
    });
});

messageInput.addEventListener("input", autoResizeInput);
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

if (newConversationButton) {
    newConversationButton.addEventListener("click", createNewConversation);
}

messageInput.focus();
loadConversationHistory();
