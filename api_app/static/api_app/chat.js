const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const typingIndicator = document.getElementById("typingIndicator");
const promptButtons = document.querySelectorAll(".prompt-btn");
const conversationList = document.getElementById("conversationList");
const newConversationButton = document.getElementById("newConversationButton");
const conversationTitle = document.getElementById("conversationTitle");
const appContainer = document.querySelector(".app-container");
const sidebarToggleButton = document.getElementById("sidebarToggleButton");
const sidebarCloseButton = document.getElementById("sidebarCloseButton");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const serviceStatus = document.getElementById("serviceStatus");
const messageCounter = document.getElementById("messageCounter");
const historyCount = document.getElementById("historyCount");

const API_URL = "/api/chat/";
const CONVERSATIONS_API_URL = "/api/chat/conversations/";
const HEALTH_URL = "/api/health/ready/";
const MESSAGE_MAX_LENGTH = 2000;
const WELCOME_PROMPTS = [
    "Comment obtenir mon attestation sur UC@Student ?",
    "Comment utiliser la plateforme PEDOC ?",
    "A quoi sert la plateforme UCAPLAT ?",
];
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
    return String(value || "")
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
    chatMessages.replaceChildren();
}

function hasWelcomeState() {
    return Boolean(chatMessages.querySelector(".chat-welcome"));
}

function openSidebar() {
    if (appContainer) {
        appContainer.classList.add("sidebar-open");
    }
}

function closeSidebar() {
    if (appContainer) {
        appContainer.classList.remove("sidebar-open");
    }
}

function createTextNode(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) {
        node.className = className;
    }
    node.textContent = text || "";
    return node;
}

function updateMessageCounter() {
    if (!messageCounter || !messageInput) {
        return;
    }
    const length = messageInput.value.length;
    messageCounter.textContent = `${length}/${MESSAGE_MAX_LENGTH}`;
    messageCounter.classList.toggle("is-near-limit", length >= 1800 && length < MESSAGE_MAX_LENGTH);
    messageCounter.classList.toggle("is-at-limit", length >= MESSAGE_MAX_LENGTH);
}

function setServiceStatus(label, stateClass) {
    if (!serviceStatus) {
        return;
    }
    serviceStatus.textContent = label;
    serviceStatus.className = `uca-pill ${stateClass || ""}`.trim();
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
        return "";
    }
    return preview.length > 22 ? `${preview.slice(0, 22)}...` : preview;
}

function formatConversationDate(value) {
    if (!value) {
        return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return "";
    }
    return new Intl.DateTimeFormat("fr-FR", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
    }).format(date);
}

function renderConversationList(conversations = []) {
    if (!conversationList) {
        return;
    }
    const conversationItems = Array.isArray(conversations) ? conversations : [];
    if (historyCount) {
        historyCount.textContent = String(conversationItems.length);
    }
    conversationList.replaceChildren();
    if (!conversationItems.length) {
        const empty = document.createElement("div");
        empty.className = "conversation-empty";
        empty.append(
            createTextNode("strong", "", "Aucune conversation"),
            createTextNode("span", "", "Cliquez sur Nouvelle conversation ou posez directement une question.")
        );
        conversationList.appendChild(empty);
        return;
    }

    conversationItems.forEach((item) => {
        const conversationId = Number(item.id || "0");
        const countLabel = `${item.message_count || 0} msg`;
        const dateLabel = formatConversationDate(item.updated_at);
        const wrapper = document.createElement("div");
        wrapper.className = item.selected ? "conversation-item is-active" : "conversation-item";
        wrapper.dataset.conversationId = String(conversationId);

        const mainButton = document.createElement("button");
        mainButton.type = "button";
        mainButton.className = "conversation-main";

        const footer = document.createElement("span");
        footer.className = "conversation-item-footer";
        footer.append(
            createTextNode("span", "conversation-item-meta", countLabel),
            createTextNode("span", "conversation-item-date", dateLabel)
        );

        mainButton.append(
            createTextNode("span", "conversation-item-title", item.title || "Nouvelle conversation"),
            createTextNode("span", "conversation-item-preview", buildConversationPreview(item)),
            footer
        );
        mainButton.addEventListener("click", async () => {
            if (!conversationId || conversationId === currentConversationId || sendButton.disabled) {
                return;
            }
            await loadConversationHistory(conversationId);
            closeSidebar();
        });

        const actions = document.createElement("div");
        actions.className = "conversation-actions";

        const renameButton = document.createElement("button");
        renameButton.type = "button";
        renameButton.className = "conversation-action-btn";
        renameButton.title = "Renommer";
        renameButton.setAttribute("aria-label", "Renommer la conversation");
        renameButton.textContent = "Renommer";
        renameButton.addEventListener("click", async (event) => {
            event.stopPropagation();
            if (!conversationId) {
                return;
            }
            const currentTitle = item.title || "Nouvelle conversation";
            const nextTitle = window.prompt("Nouveau titre de la conversation", currentTitle.trim());
            if (!nextTitle || !nextTitle.trim()) {
                return;
            }
            await updateConversation(conversationId, { title: nextTitle.trim() });
        });

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "conversation-action-btn conversation-action-danger";
        deleteButton.title = "Supprimer";
        deleteButton.setAttribute("aria-label", "Supprimer la conversation");
        deleteButton.textContent = "Supprimer";
        deleteButton.addEventListener("click", async (event) => {
            event.stopPropagation();
            if (!conversationId) {
                return;
            }
            const confirmed = window.confirm("Supprimer cette conversation de votre historique actif ?");
            if (!confirmed) {
                return;
            }
            await deleteConversation(conversationId);
        });

        actions.append(renameButton, deleteButton);
        wrapper.append(mainButton, actions);
        conversationList.appendChild(wrapper);
    });
}

function appendSourceList(content, sources) {
    if (!Array.isArray(sources) || !sources.length) {
        return;
    }
    const shell = document.createElement("div");
    shell.className = "message-sources";
    shell.appendChild(createTextNode("p", "", "Sources utiles"));
    const list = document.createElement("ul");
    sources.slice(0, 3).forEach((source) => {
        const item = document.createElement("li");
        item.appendChild(createTextNode("span", "", source.name || source.path || "Source"));
        list.appendChild(item);
    });
    shell.appendChild(list);
    content.appendChild(shell);
}

function appendMessage(role, text, options = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "message-user" : "message-assistant"}`;

    if (role !== "user") {
        const avatar = document.createElement("div");
        avatar.className = "avatar avatar-ai";
        avatar.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 100 20 10 10 0 000-20zM12 18a6 6 0 100-12 6 6 0 000 12z"></path></svg>`;
        wrapper.appendChild(avatar);
    }

    const content = document.createElement("div");
    content.className = "message-content";

    const paragraph = document.createElement("p");
    paragraph.textContent = text || "";
    content.appendChild(paragraph);

    if (role !== "user") {
        if (options.confidence) {
            content.appendChild(createTextNode("div", "message-meta", `Confiance: ${options.confidence}`));
        }
        appendSourceList(content, options.sources || []);

        if (options.id) {
            // --- Rating row ---
            const feedbackWidget = document.createElement("div");
            feedbackWidget.className = "message-feedback-widget";

            const upBtn = document.createElement("button");
            upBtn.type = "button";
            upBtn.className = "feedback-btn btn-up";
            upBtn.innerHTML = "👍";
            upBtn.title = "Réponse utile";

            const downBtn = document.createElement("button");
            downBtn.type = "button";
            downBtn.className = "feedback-btn btn-down";
            downBtn.innerHTML = "👎";
            downBtn.title = "Réponse non utile";

            const thanksSpan = document.createElement("span");
            thanksSpan.className = "feedback-thanks";
            thanksSpan.textContent = "Merci pour votre retour !";

            feedbackWidget.append(upBtn, downBtn, thanksSpan);

            // --- Comment row (hidden by default) ---
            const commentRow = document.createElement("div");
            commentRow.className = "feedback-comment-row";

            const commentInput = document.createElement("textarea");
            commentInput.className = "feedback-comment-input";
            commentInput.placeholder = "Ajoutez un commentaire (optionnel)…";
            commentInput.rows = 2;
            commentInput.maxLength = 500;

            const sendCommentBtn = document.createElement("button");
            sendCommentBtn.type = "button";
            sendCommentBtn.className = "feedback-comment-send";
            sendCommentBtn.textContent = "Envoyer";

            commentRow.append(commentInput, sendCommentBtn);

            // --- Restore existing feedback on history load ---
            let currentRating = null;
            if (options.feedback) {
                currentRating = options.feedback.rating;
                if (currentRating === "up") {
                    upBtn.classList.add("is-active");
                } else if (currentRating === "down") {
                    downBtn.classList.add("is-active");
                }
                if (options.feedback.comment) {
                    commentInput.value = options.feedback.comment;
                    commentInput.readOnly = true;
                    sendCommentBtn.style.display = "none";
                    commentRow.classList.add("is-visible");
                }
            }

            // --- Submit rating + show comment box ---
            const submitRating = async (rating) => {
                try {
                    const csrfToken = getCsrfToken();
                    const res = await fetch(`/api/chat/messages/${options.id}/feedback/`, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-CSRFToken": csrfToken,
                        },
                        body: JSON.stringify({ rating }),
                    });
                    if (res.ok) {
                        currentRating = rating;
                        upBtn.classList.toggle("is-active", rating === "up");
                        upBtn.classList.toggle("is-dimmed", rating !== "up");
                        downBtn.classList.toggle("is-active", rating === "down");
                        downBtn.classList.toggle("is-dimmed", rating !== "down");
                        // Show comment field
                        commentInput.readOnly = false;
                        sendCommentBtn.style.display = "";
                        commentRow.classList.add("is-visible");
                        commentInput.focus();
                    }
                } catch (err) {
                    console.error("Feedback submission failed", err);
                }
            };

            // --- Submit comment ---
            const submitComment = async () => {
                const comment = commentInput.value.trim();
                if (!currentRating) return;
                try {
                    const csrfToken = getCsrfToken();
                    const res = await fetch(`/api/chat/messages/${options.id}/feedback/`, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-CSRFToken": csrfToken,
                        },
                        body: JSON.stringify({ rating: currentRating, comment }),
                    });
                    if (res.ok) {
                        commentInput.readOnly = true;
                        sendCommentBtn.style.display = "none";
                        thanksSpan.classList.add("is-visible");
                        setTimeout(() => thanksSpan.classList.remove("is-visible"), 3000);
                    }
                } catch (err) {
                    console.error("Comment submission failed", err);
                }
            };

            upBtn.addEventListener("click", () => submitRating("up"));
            downBtn.addEventListener("click", () => submitRating("down"));
            sendCommentBtn.addEventListener("click", submitComment);
            commentInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submitComment();
                }
            });

            content.appendChild(feedbackWidget);
            content.appendChild(commentRow);
        }
    }

    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function renderWelcomeState() {
    clearInitialMessages();

    const shell = document.createElement("section");
    shell.className = "chat-welcome";

    const logo = document.createElement("img");
    logo.className = "chat-welcome-logo";
    logo.src = "/static/api_app/img/logo_uca.webp";
    logo.alt = "Universite Cadi Ayyad";

    const title = createTextNode("h3", "chat-welcome-title", "Comment puis-je vous aider ?");
    const subtitle = createTextNode(
        "p",
        "chat-welcome-subtitle",
        "Posez une question sur les services numeriques, la scolarite ou les procedures UCA."
    );

    const prompts = document.createElement("div");
    prompts.className = "chat-welcome-prompts";

    WELCOME_PROMPTS.forEach((prompt) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "chat-welcome-prompt";
        button.textContent = prompt;
        button.addEventListener("click", async () => {
            if (sendButton.disabled) {
                return;
            }
            messageInput.value = prompt;
            autoResizeInput();
            updateMessageCounter();
            await submitMessage(prompt);
        });
        prompts.appendChild(button);
    });

    shell.append(logo, title, subtitle, prompts);
    chatMessages.appendChild(shell);
    scrollToBottom();
}

function setLoadingState(isLoading, showTypingIndicator = false) {
    sendButton.disabled = isLoading;
    messageInput.disabled = isLoading;
    typingIndicator.hidden = !showTypingIndicator;
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
    if (hasWelcomeState()) {
        clearInitialMessages();
    }
    appendMessage("user", message);
    messageInput.value = "";
    autoResizeInput();
    updateMessageCounter();
    setLoadingState(true, true);

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
            id: payload.message_id,
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
            renderWelcomeState();
        } else {
            messages.forEach((message) => {
                appendMessage(message.role, message.content || "", {
                    id: message.id,
                    confidence: message.confidence || "",
                    sources: message.sources || [],
                    feedback: message.feedback || null,
                });
            });
        }
    } catch (error) {
        appendMessage("assistant", "Je n'ai pas pu charger l'historique de cette conversation pour le moment.");
    } finally {
        setLoadingState(false);
    }
}

async function loadServiceStatus() {
    setServiceStatus("Verification service", "uca-pill-warn");
    try {
        const response = await fetch(HEALTH_URL, {
            method: "GET",
            credentials: "same-origin",
            headers: { "X-Requested-With": "XMLHttpRequest" },
        });
        if (response.ok) {
            setServiceStatus("Service pret", "uca-pill-ok");
            return;
        }
        setServiceStatus("Service a verifier", "uca-pill-warn");
    } catch (error) {
        setServiceStatus("Service indisponible", "uca-pill-bad");
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
        renderWelcomeState();
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

async function deleteConversation(conversationId) {
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
            throw new Error("Suppression impossible.");
        }
        const data = await response.json();
        renderConversationList(data.conversations || []);
        if (currentConversationId === conversationId) {
            currentConversationId = null;
            shouldOpenFreshConversation = true;
        }
    } catch (error) {
        appendMessage("assistant", `Je n'ai pas pu supprimer cette conversation.\nDetail: ${error.message || "Erreur reseau."}`);
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
        updateMessageCounter();
        await submitMessage(prompt);
        closeSidebar();
    });
});

messageInput.addEventListener("input", () => {
    autoResizeInput();
    updateMessageCounter();
});
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

if (newConversationButton) {
    newConversationButton.addEventListener("click", createNewConversation);
}

if (sidebarToggleButton) {
    sidebarToggleButton.addEventListener("click", openSidebar);
}

if (sidebarCloseButton) {
    sidebarCloseButton.addEventListener("click", closeSidebar);
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener("click", closeSidebar);
}

messageInput.focus();
updateMessageCounter();
loadServiceStatus();
loadConversationHistory();
