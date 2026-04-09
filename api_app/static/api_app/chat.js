const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const typingIndicator = document.getElementById("typingIndicator");
const promptButtons = document.querySelectorAll(".prompt-btn");

const API_URL = "/api/chat/";

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

function appendMessage(role, text) {
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

    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function setLoadingState(isLoading) {
    sendButton.disabled = isLoading;
    messageInput.disabled = isLoading;
    typingIndicator.hidden = !isLoading;
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
        appendMessage("assistant", answer);
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

messageInput.focus();
