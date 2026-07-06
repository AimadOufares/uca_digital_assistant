function getCookie(name) {
    const cookies = document.cookie ? document.cookie.split("; ") : [];
    for (const cookie of cookies) {
        const [key, ...rest] = cookie.split("=");
        if (key === name) return decodeURIComponent(rest.join("="));
    }
    return "";
}

function byId(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const node = byId(id);
    if (node) node.textContent = value;
}

function clearNode(node) {
    while (node && node.firstChild) {
        node.removeChild(node.firstChild);
    }
}

function percent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function fixed(value, digits = 2) {
    return Number(value || 0).toFixed(digits);
}

function formatDate(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "-";
    return date.toLocaleString("fr-FR");
}

function formatSizeKb(value) {
    return `${Number(value || 0).toFixed(2)} KB`;
}

function badgeClassFromStatus(value) {
    if (value === "up_to_date" || value === "ok") return "ok";
    if (value === "rebuild_required" || value === "degraded") return "warn";
    return "bad";
}

function readyBadgeClass(payload) {
    if (payload?.ready) return "status-badge status-ok";
    if (payload?.llm?.state === "degraded") return "status-badge status-warn";
    return "status-badge status-bad";
}

function setActionStatus(message, isError = false) {
    const box = byId("drive-action-status");
    if (!box) return;
    box.textContent = message;
    box.classList.toggle("status-bad", Boolean(isError));
}

function renderProcessingSummary(processing) {
    const box = byId("drive-processing-summary");
    if (!box) return;
    clearNode(box);

    const summaries = processing?.corpora || [];
    if (!summaries.length) {
        box.hidden = true;
        return;
    }

    const title = document.createElement("strong");
    title.textContent = "Resume du processing incremental";
    box.appendChild(title);

    summaries.forEach((item) => {
        const row = document.createElement("p");
        row.textContent = [
            `${item.corpus || "corpus"}:`,
            `${item.detected || 0} detecte(s)`,
            `${item.processed || 0} traite(s)`,
            `${item.skipped_unchanged || 0} ignore(s)`,
            `${item.skipped_no_chunks || 0} sans chunk`,
            `${item.failed || 0} erreur(s)`,
            `${item.quarantined || 0} quarantaine`,
        ].join(" ");
        box.appendChild(row);
    });
    box.hidden = false;
}

function makeCell(text, tagName = "td") {
    const cell = document.createElement(tagName);
    cell.textContent = text ?? "-";
    return cell;
}

function makeEmptyRow(colspan, message) {
    const row = document.createElement("tr");
    const cell = makeCell(message);
    cell.colSpan = colspan;
    cell.className = "empty";
    row.appendChild(cell);
    return row;
}

function makePill(text, klass) {
    const pill = document.createElement("span");
    pill.className = `pill ${klass || ""}`.trim();
    pill.textContent = text || "-";
    return pill;
}

function renderKeyValueList(id, rows) {
    const list = byId(id);
    if (!list) return;
    clearNode(list);
    rows.forEach(([label, value]) => {
        const row = document.createElement("div");
        row.className = "metric-row";
        const left = document.createElement("span");
        const right = document.createElement("strong");
        left.textContent = label;
        right.textContent = value;
        row.append(left, right);
        list.appendChild(row);
    });
}

async function loadDashboard() {
    try {
        const response = await fetch("/api/dashboard-metrics/");
        if (response.status === 403) {
            throw new Error("Acces reserve a un administrateur connecte.");
        }
        if (!response.ok) {
            throw new Error("Impossible de charger les metriques.");
        }
        const payload = await response.json();
        renderDashboard(payload);
        hideLoader();
        loadDriveDocuments();
        loadConversationAudit();
    } catch (error) {
        renderLoaderError(error.message || "Chargement impossible.");
    }
}

function renderLoaderError(message) {
    const loading = byId("loading");
    if (!loading) return;
    clearNode(loading);
    const card = document.createElement("div");
    card.className = "loading-card";
    const title = document.createElement("h2");
    const detail = document.createElement("p");
    title.textContent = "Chargement impossible";
    detail.textContent = message;
    card.append(title, detail);
    loading.appendChild(card);
}

function hideLoader() {
    const loading = byId("loading");
    if (!loading) return;
    loading.classList.add("hidden");
    setTimeout(() => {
        loading.style.display = "none";
    }, 220);
}

function renderDashboard(payload) {
    const system = payload.system_status || {};
    const activeIndex = payload.active_index || {};
    const driveSync = payload.drive_sync_status || {};
    const llmConfig = payload.llm_config || {};
    const audit = payload.data_audit || {};
    const quality = payload.raw_quality_audit || {};
    const evaluation = payload.rag_eval || {};
    const latestReports = payload.latest_reports || {};

    const readyBadge = byId("ready-badge");
    if (readyBadge) readyBadge.className = readyBadgeClass(system);
    setText("ready-label", system.ready ? "Systeme pret" : "Systeme a verifier");
    setText("meta-llm", llmConfig.chat_model || "-");
    setText("meta-embedding", activeIndex.embedding_model || "-");
    setText("meta-build", activeIndex.build_id || "-");
    setText("meta-sync", driveSync.status || "-");

    setText("kpi-ready", system.ready ? "Oui" : "Non");
    setText("kpi-ready-sub", system.ready ? "Service disponible pour les reponses." : "Verification requise.");
    setText("kpi-build", activeIndex.build_id || "-");
    setText("kpi-build-sub", activeIndex.manifest_updated_at ? `Publie le ${formatDate(activeIndex.manifest_updated_at)}` : "Aucun build publie detecte.");
    setText("kpi-chunks", String(activeIndex.chunk_count || 0));
    setText("kpi-sources", String(activeIndex.source_count || 0));
    setText("kpi-drive-docs", String(driveSync.document_count || 0));
    if (evaluation.summary) {
        setText("kpi-benchmark", evaluation.benchmark || "drive");
        setText("kpi-benchmark-sub", `${evaluation.questions_evaluated || 0} questions evaluees`);
    } else {
        setText("kpi-benchmark", "-");
        setText("kpi-benchmark-sub", "Aucun benchmark charge.");
    }
    setText("admin-index-note", activeIndex.build_id ? `Build ${activeIndex.build_id} - ${activeIndex.embedding_model || "-"}` : "Aucun index publie detecte.");

    renderDriveSync(driveSync);

    setText("quality-clean", String(quality.quality?.documents_without_flags || 0));
    setText("quality-flagged", String(quality.quality?.documents_with_any_flag || 0));
    setText("quality-raw-files", String(audit.raw?.files || 0));
    setText("quality-raw-size", `${audit.raw?.size_mb || 0} MB de fichiers bruts`);

    const serviceDistribution = payload.data_audit?.index_metadata?.service_name_distribution || {};
    const rankedServices = Object.entries(serviceDistribution)
        .sort((a, b) => b[1] - a[1])
        .filter((item) => item[0] && item[0] !== "unknown");
    setText("quality-services", String(rankedServices.length));
    renderServicesTable(rankedServices);

    renderAuditSummary(latestReports);
    renderEvaluation(evaluation);
}

function renderDriveSync(driveSync) {
    const box = byId("drive-sync-box");
    if (!box) return;
    clearNode(box);

    const strong = document.createElement("strong");
    strong.textContent = "Synchronisation du corpus";
    const pill = makePill(driveSync.status || "unknown", badgeClassFromStatus(driveSync.status));
    const latest = document.createElement("p");
    const active = document.createElement("p");
    latest.textContent = `Dernier document : ${formatDate(driveSync.latest_document_updated_at)}`;
    active.textContent = `Index actif : ${formatDate(driveSync.active_index_updated_at)}`;

    box.append(strong, document.createElement("br"), pill);
    if (driveSync.latest_document_name) {
        const doc = document.createElement("p");
        doc.textContent = `Document modifie : ${driveSync.latest_document_name}`;
        box.appendChild(doc);
    }
    box.append(latest, active);
}

function renderServicesTable(items) {
    const body = byId("services-table-body");
    if (!body) return;
    clearNode(body);
    if (!items.length) {
        body.appendChild(makeEmptyRow(2, "Aucune distribution service disponible."));
        return;
    }
    items.slice(0, 8).forEach(([name, count]) => {
        const row = document.createElement("tr");
        row.append(makeCell(name), makeCell(String(count)));
        body.appendChild(row);
    });
}

function renderAuditSummary(latestReports) {
    const rows = [
        ["Audit de donnees", latestReports.data_audit],
        ["Audit raw quality", latestReports.raw_quality_audit],
        ["Benchmark drive", latestReports.rag_eval],
    ].map(([label, bundle]) => [
        label,
        bundle?.available ? formatDate(bundle.updated_at) : "Non disponible",
    ]);
    renderKeyValueList("audit-summary-list", rows);
}

function renderEvaluation(evaluation) {
    const body = byId("eval-table-body");
    if (!evaluation || !evaluation.summary) {
        setText("eval-summary-box", "Aucun benchmark drive disponible.");
        setText("eval-service-accuracy", "-");
        setText("eval-best-match", "-");
        setText("eval-abstention", "-");
        setText("eval-latency", "-");
        if (body) {
            clearNode(body);
            body.appendChild(makeEmptyRow(5, "Aucun benchmark charge."));
        }
        return;
    }
    const summary = evaluation.summary;
    setText("eval-summary-box", `Benchmark ${evaluation.benchmark || "drive"} - ${evaluation.questions_evaluated || 0} questions evaluees`);
    setText("eval-service-accuracy", percent(summary.service_top1_accuracy));
    setText("eval-best-match", percent(summary.best_match_score_avg));
    setText("eval-abstention", percent(summary.abstention_rate));
    setText("eval-latency", `${fixed(summary.retrieval_latency_ms_avg, 0)} ms`);

    if (!body) return;
    clearNode(body);
    const rows = evaluation.rows || [];
    if (!rows.length) {
        body.appendChild(makeEmptyRow(5, "Aucun cas evalue."));
    }
}

function activateSection(sectionId) {
    const sections = document.querySelectorAll('.admin-shell .admin-section');
    sections.forEach((section) => section.classList.toggle('active', section.dataset.section === sectionId));
}

function activateSidebarLink(link) {
    const links = document.querySelectorAll('.sidebar-link');
    links.forEach((item) => item.classList.toggle('active', item === link));
    const sectionId = link.dataset.section || link.getAttribute('href')?.slice(1);
    if (sectionId) {
        activateSection(sectionId);
        window.history.replaceState(null, '', `#${sectionId}`);
    }
}

function setupSidebarNavigation() {
    const links = Array.from(document.querySelectorAll('.sidebar-link'));
    if (!links.length) return;

    const activateTarget = (hash) => {
        const target = links.find((item) => item.getAttribute('href') === hash);
        if (target) {
            activateSidebarLink(target);
        }
    };

    links.forEach((link) => {
        link.addEventListener('click', () => {
            activateSidebarLink(link);
        });
    });

    if (window.location.hash) {
        activateTarget(window.location.hash);
    } else {
        activateSidebarLink(links[0]);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    setupSidebarNavigation();
    loadDashboard();
});

async function loadDriveDocuments() {
    const body = byId("drive-doc-table-body");
    if (!body) return;
    clearNode(body);
    try {
        const response = await fetch("/api/drive-documents/");
        if (response.status === 403) {
            body.appendChild(makeEmptyRow(4, "Acces refuse."));
            return;
        }
        const payload = await response.json();
        const documents = payload.documents || [];
        setText("drive-doc-count", `${payload.count || 0} document(s)`);
        if (!documents.length) {
            body.appendChild(makeEmptyRow(4, "Aucun document dans le corpus drive."));
            return;
        }
        documents.forEach((doc) => {
            const row = document.createElement("tr");
            row.append(
                makeCell(doc.name || "-"),
                makeCell(formatSizeKb(doc.size_kb)),
                makeCell(formatDate(Number(doc.updated_at || 0) * 1000))
            );
            const actionCell = document.createElement("td");
            const button = document.createElement("button");
            button.className = "uca-btn uca-btn-danger";
            button.type = "button";
            button.textContent = "Supprimer";
            button.addEventListener("click", () => deleteDriveDocument(doc.name || ""));
            actionCell.appendChild(button);
            row.appendChild(actionCell);
            body.appendChild(row);
        });
    } catch (error) {
        body.appendChild(makeEmptyRow(4, "Erreur de chargement des documents."));
    }
}

async function uploadDriveDocument() {
    const input = byId("drive-file-input");
    const file = input?.files?.[0];
    if (!file) {
        setActionStatus("Selectionne un document avant l'ajout.", true);
        return;
    }
    const formData = new FormData();
    formData.append("file", file);
    setActionStatus("Ajout du document en cours...");
    try {
        const response = await fetch("/api/drive-documents/", {
            method: "POST",
            headers: { "X-CSRFToken": getCookie("csrftoken") },
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Echec de l'ajout.");
        input.value = "";
        setActionStatus(`${payload.detail} Rebuild requis pour publier le nouvel index.`);
        await refreshDashboard();
    } catch (error) {
        setActionStatus(error.message || "Erreur pendant l'ajout.", true);
    }
}

async function deleteDriveDocument(filename) {
    if (!filename) return;
    if (!window.confirm(`Supprimer ${filename} du corpus drive ?`)) return;
    setActionStatus(`Suppression de ${filename}...`);
    try {
        const encodedName = encodeURIComponent(filename);
        const response = await fetch(`/api/drive-documents/${encodedName}/`, {
            method: "DELETE",
            headers: { "X-CSRFToken": getCookie("csrftoken") },
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Suppression impossible.");
        setActionStatus(`${payload.detail} Rebuild requis pour mettre a jour l'index.`);
        await refreshDashboard();
    } catch (error) {
        setActionStatus(error.message || "Erreur pendant la suppression.", true);
    }
}

async function rebuildDrive() {
    setActionStatus("Rebuild drive en cours : processing + indexation publiee...");
    renderProcessingSummary(null);
    try {
        const response = await fetch("/api/drive-rebuild/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Echec du rebuild drive.");
        const processing = payload.processing?.corpora?.[0] || {};
        setActionStatus(
            `Rebuild termine. Build publie : ${payload.index?.build_id || "-"}. ` +
            `${processing.processed || 0} document(s) traite(s), ${processing.skipped_unchanged || 0} ignore(s).`
        );
        renderProcessingSummary(payload.processing);
        await refreshDashboard();
    } catch (error) {
        setActionStatus(error.message || "Erreur pendant le rebuild.", true);
        renderProcessingSummary(null);
    }
}

async function evaluateDrive() {
    setActionStatus("Benchmark drive en cours...");
    try {
        const response = await fetch("/api/drive-evaluate/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Echec du benchmark drive.");
        setActionStatus(payload.detail || "Benchmark drive regenere.");
        await refreshDashboard();
    } catch (error) {
        setActionStatus(error.message || "Erreur pendant le benchmark.", true);
    }
}

async function refreshDashboard() {
    const response = await fetch("/api/dashboard-metrics/");
    const payload = await response.json();
    renderDashboard(payload);
    await loadDriveDocuments();
    await loadConversationAudit();
}

function bindAction(id, handler) {
    const node = byId(id);
    if (node) node.addEventListener("click", handler);
}

function appendAuditLog(message, isError = false) {
    const consoleBox = byId("audit-console");
    if (!consoleBox) return;
    const line = document.createElement("div");
    line.className = isError ? "audit-log-line is-error" : "audit-log-line";
    line.textContent = `[${new Date().toLocaleTimeString("fr-FR")}] ${message}`;
    if (consoleBox.textContent === "Aucun audit lance.") {
        clearNode(consoleBox);
    }
    consoleBox.appendChild(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

async function runAuditTask(task, button) {
    if (!task || !button) return;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "En cours...";
    appendAuditLog(`Lancement ${originalText}`);
    try {
        const response = await fetch(`/api/run-audit/${encodeURIComponent(task)}/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({}),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || payload.error || "Audit impossible.");
        }
        appendAuditLog(`${payload.label || originalText} termine en ${payload.elapsed_s || 0}s`);
        await refreshDashboard();
    } catch (error) {
        appendAuditLog(`${originalText}: ${error.message || "erreur inconnue"}`, true);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

function bindAuditActions() {
    document.querySelectorAll("[data-audit-task]").forEach((button) => {
        button.addEventListener("click", () => runAuditTask(button.dataset.auditTask, button));
    });
}

async function loadConversationAudit() {
    const body = byId("conversation-audit-table-body");
    if (!body) return;
    try {
        const response = await fetch("/api/admin-conversations/");
        if (response.status === 403) {
            body.replaceChildren(makeEmptyRow(6, "Acces refuse."));
            return;
        }
        if (!response.ok) {
            throw new Error("Chargement impossible.");
        }
        const payload = await response.json();
        const summary = payload.summary || {};
        setText("conv-active", String(summary.active_conversations || 0));
        setText("conv-messages", String(summary.total_messages || 0));
        setText("conv-answers", String(summary.assistant_answers || 0));
        setText("conv-source-coverage", percent(summary.source_coverage));

        clearNode(body);
        const conversations = payload.recent || [];
        if (!conversations.length) {
            body.appendChild(makeEmptyRow(6, "Aucune conversation active."));
            return;
        }
        conversations.forEach((item) => {
            const row = document.createElement("tr");
            const sourceLabel = `${item.source_answer_count || 0}/${item.assistant_count || 0}`;
            row.append(
                makeCell(item.user || "-"),
                makeCell(item.title || "-"),
                makeCell(String(item.message_count || 0)),
                makeCell(sourceLabel),
                makeCell(formatDate(item.last_message_at)),
                makeCell(item.preview || "-")
            );
            body.appendChild(row);
        });
    } catch (error) {
        clearNode(body);
        body.appendChild(makeEmptyRow(6, "Erreur de chargement des conversations."));
    }
}

bindAction("upload-drive-btn", uploadDriveDocument);
bindAction("rebuild-drive-btn", rebuildDrive);
bindAction("rebuild-drive-btn-bottom", rebuildDrive);
bindAction("evaluate-drive-btn", evaluateDrive);
bindAction("evaluate-drive-btn-bottom", evaluateDrive);
bindAuditActions();
