const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const typingIndicator = document.getElementById("typingIndicator");
const promptChips = document.querySelectorAll(".prompt-chip");

const API_URL = "/api/chat/";

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function autoResizeInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 140)}px`;
}

function sourceLabel(source) {
    if (typeof source === "string") {
        return source;
    }
    if (!source || typeof source !== "object") {
        return "Document";
    }

    const base = source.name || source.path || "Document";
    const score = typeof source.score === "number" ? `score ${source.score.toFixed(3)}` : "";
    const scoreType = typeof source.score_type === "string" ? source.score_type : "";
    const hits = typeof source.hits === "number" && source.hits > 1 ? `${source.hits} extraits` : "";

    const extras = [score, scoreType, hits].filter(Boolean).join(" • ");
    return extras ? `${base} (${extras})` : base;
}

function appendMessage(role, text, sources = []) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${role === "user" ? "message-user" : "message-assistant"}`;

    if (role !== "user") {
        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = "UCA";
        wrapper.appendChild(avatar);
    }

    const body = document.createElement("div");
    body.className = "message-body";

    const paragraph = document.createElement("p");
    paragraph.innerHTML = escapeHtml(text).replaceAll("\n", "<br>");
    body.appendChild(paragraph);

    if (Array.isArray(sources) && sources.length > 0) {
        const sourcesBlock = document.createElement("div");
        sourcesBlock.className = "sources";

        const title = document.createElement("strong");
        title.textContent = "Sources";
        sourcesBlock.appendChild(title);

        const list = document.createElement("ul");
        sources.forEach((source) => {
            const item = document.createElement("li");
            item.textContent = sourceLabel(source);
            list.appendChild(item);
        });
        sourcesBlock.appendChild(list);
        body.appendChild(sourcesBlock);
    }

    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function setLoadingState(isLoading) {
    sendButton.disabled = isLoading;
    messageInput.disabled = isLoading;
    typingIndicator.hidden = !isLoading;
    promptChips.forEach((chip) => {
        chip.disabled = isLoading;
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
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
            body: JSON.stringify({ message }),
        });

        let payload = {};
        try {
            payload = await response.json();
        } catch (error) {
            payload = {};
        }

        if (!response.ok) {
            const detail = payload.detail || "Le serveur a retourne une erreur.";
            throw new Error(detail);
        }

        const answer = payload.answer || "Aucune reponse generee.";
        const sources = Array.isArray(payload.sources) ? payload.sources : [];
        appendMessage("assistant", answer, sources);
    } catch (error) {
        appendMessage(
            "assistant",
            `Je n'ai pas pu traiter votre demande pour le moment.\nDetail: ${error.message || "Erreur reseau."}`
        );
    } finally {
        setLoadingState(false);
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

promptChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
        const prompt = (chip.dataset.prompt || "").trim();
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

messageInput.focus();
