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
    // Show status in the Corpus section panel
    const box = byId("drive-action-status");
    if (box) {
        box.textContent = message;
        box.classList.toggle("status-bad", Boolean(isError));
    }
    // Also mirror in the Maintenance section status area
    const maintBox = byId("maint-action-status");
    const maintText = byId("maint-action-status-text");
    const maintPulse = maintBox ? maintBox.querySelector(".status-dot-pulse") : null;
    
    if (maintBox && maintText) {
        maintBox.style.display = "block";
        maintText.textContent = `[${new Date().toLocaleTimeString("fr-FR")}] ${message}`;
        
        if (maintPulse) {
            if (isError) {
                maintPulse.style.backgroundColor = "var(--uca-danger)";
            } else if (message.includes("en cours") || message.includes("cours...")) {
                maintPulse.style.backgroundColor = "var(--uca-blue)";
            } else {
                maintPulse.style.backgroundColor = "#137333"; // Success green
            }
        }
    }
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
    
    // Meta Grid - Enhanced LLM info with state and provider
    const llm = system.llm || {};
    const llmState = llm.state || "unknown";
    const llmProvider = llmConfig.provider || "unknown";
    const llmModelWithState = `${llmConfig.chat_model || "-"} (${llmState})`;
    setText("meta-llm", llmModelWithState);
    
    setText("meta-embedding", activeIndex.embedding_model || "-");
    
    // Build with timestamp
    const buildDate = activeIndex.manifest_updated_at 
        ? formatDate(activeIndex.manifest_updated_at).split(' ').slice(0, 2).join(' ')
        : "";
    const buildDisplay = activeIndex.build_id ? `${activeIndex.build_id} ${buildDate ? '(' + buildDate + ')' : ''}` : "-";
    setText("meta-build", buildDisplay.trim());
    
    setText("meta-sync", driveSync.status || "-");

    // KPI Grid - System Ready with Check Details
    const checks = system.checks || {};
    const systemStatus = system.ready ? "✓ Operationnel" : "⚠ Non pret";
    setText("kpi-ready", systemStatus);
    setText("kpi-ready-sub", system.ready ? "Service disponible pour les reponses." : "Verification requise.");
    
    // Show check details
    const checksBox = byId("kpi-ready-checks");
    if (checksBox) {
        checksBox.style.display = "block";
        const dbStatus = checks.database_ready ? "✓ DB" : "✗ DB";
        const vectorStatus = checks.vector_store_ready ? "✓ Index" : "✗ Index";
        const llmStatus = checks.llm_ready ? "✓ LLM" : "✗ LLM";
        setText("kpi-check-db", dbStatus);
        setText("kpi-check-vector", vectorStatus);
        setText("kpi-check-llm", llmStatus);
    }
    
    // KPI Grid - Index Published (was "Build actif")
    setText("kpi-build", activeIndex.build_id || "-");
    setText("kpi-build-sub", activeIndex.manifest_updated_at 
        ? `Publie ${formatDate(activeIndex.manifest_updated_at)}`
        : "Aucun build publie.");
    
    // KPI Grid - Chunks and Sources (unchanged)
    setText("kpi-chunks", String(activeIndex.chunk_count || 0));
    setText("kpi-sources", String(activeIndex.source_count || 0));
    setText("kpi-drive-docs", String(driveSync.document_count || 0));
    
    setText("admin-index-build-id", activeIndex.build_id || "-");
    setText("admin-index-date", activeIndex.manifest_updated_at ? formatDate(activeIndex.manifest_updated_at) : "-");
    setText("admin-index-chunks", String(activeIndex.chunk_count || 0));
    setText("admin-index-sources", String(activeIndex.source_count || 0));
    setText("admin-index-embedding", activeIndex.embedding_model || "-");

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
    renderQualityBreakdowns(quality);

    renderAuditSummary(latestReports);
    renderEvaluation(evaluation);
}

function renderQualityBreakdowns(quality) {
    const flagsList = byId("quality-flags-list");
    if (flagsList) {
        clearNode(flagsList);
        const counts = quality.quality?.flag_counts || {};
        const translations = {
            lang_not_allowed: "Langue non autorisée",
            lang_conf_low: "Confiance linguistique faible",
            too_short_words: "Fichiers trop courts (mots)",
            too_short_chars: "Fichiers trop courts (caractères)",
            off_topic: "Hors-sujet / Hors-scope",
            quality_score_low: "Qualité de texte dégradée",
            lexical_diversity_low: "Faible diversité lexicale"
        };
        const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        if (!entries.length) {
            const row = document.createElement("div");
            row.className = "metric-row";
            row.innerHTML = "<span>Aucun signal de qualité détecté</span><strong>0</strong>";
            flagsList.appendChild(row);
        } else {
            entries.forEach(([key, count]) => {
                const label = translations[key] || key;
                const row = document.createElement("div");
                row.className = "metric-row";
                const left = document.createElement("span");
                const right = document.createElement("strong");
                left.textContent = label;
                right.textContent = String(count);
                if (count > 0) {
                    right.style.color = "var(--uca-danger)";
                }
                row.append(left, right);
                flagsList.appendChild(row);
            });
        }
    }

    const formatsList = byId("document-formats-list");
    if (formatsList) {
        clearNode(formatsList);
        
        // Formats / Extensions
        const exts = quality.summary?.extensions || {};
        Object.entries(exts).forEach(([ext, count]) => {
            const row = document.createElement("div");
            row.className = "metric-row";
            const left = document.createElement("span");
            const right = document.createElement("strong");
            left.textContent = `Format ${ext.toUpperCase()}`;
            right.textContent = `${count} fichier(s)`;
            row.append(left, right);
            formatsList.appendChild(row);
        });

        // Langues
        const langs = quality.summary?.languages || {};
        const langNames = {
            fr: "Français",
            ar: "Arabe",
            en: "Anglais",
            unknown: "Non déterminée"
        };
        Object.entries(langs).forEach(([lang, count]) => {
            const row = document.createElement("div");
            row.className = "metric-row";
            const left = document.createElement("span");
            const right = document.createElement("strong");
            left.textContent = `Langue : ${langNames[lang] || lang}`;
            right.textContent = `${count} fichier(s)`;
            row.append(left, right);
            formatsList.appendChild(row);
        });
    }
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
        setText("eval-hit-rate", "-");
        setText("eval-precision", "-");
        setText("eval-coverage", "-");
        setText("eval-abstention", "-");
        setText("eval-latency", "-");
        if (body) {
            clearNode(body);
            body.appendChild(makeEmptyRow(7, "Aucun benchmark charge."));
        }
        return;
    }
    const summary = evaluation.summary;
    setText("eval-summary-box", `Benchmark ${evaluation.benchmark || "drive"} — ${evaluation.questions_evaluated || 0} questions évaluées`);
    setText("eval-service-accuracy", percent(summary.service_top1_accuracy));
    setText("eval-hit-rate", percent(summary.hit_at_k_rate));
    setText("eval-precision", percent(summary.precision_at_k_avg));
    setText("eval-coverage", percent(summary.coverage_at_k_avg));
    setText("eval-abstention", percent(summary.abstention_rate));
    setText("eval-latency", `${fixed(summary.retrieval_latency_ms_avg, 0)} ms`);

    if (!body) return;
    clearNode(body);
    const rows = evaluation.rows || [];
    if (!rows.length) {
        body.appendChild(makeEmptyRow(8, "Aucun cas évalué."));
    } else {
        rows.forEach((row) => {
            const tr = document.createElement("tr");

            const isMatch = Number(row.service_top1_match) === 1;
            const statusPill = makePill(isMatch ? "Correct" : "Erreur", isMatch ? "ok" : "bad");
            const statusCell = document.createElement("td");
            statusCell.appendChild(statusPill);

            const actionCell = document.createElement("td");
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "uca-btn";
            deleteBtn.style.padding = "4px 8px";
            deleteBtn.style.fontSize = "11px";
            deleteBtn.style.borderRadius = "var(--uca-radius)";
            deleteBtn.style.border = "1px solid var(--uca-danger)";
            deleteBtn.style.color = "var(--uca-danger)";
            deleteBtn.style.background = "transparent";
            deleteBtn.style.cursor = "pointer";
            deleteBtn.textContent = "Supprimer";
            deleteBtn.addEventListener("click", async () => {
                if (confirm(`Voulez-vous vraiment supprimer la question : "${row.question}" ?`)) {
                    deleteBtn.disabled = true;
                    try {
                        const response = await fetch("/api/benchmark/delete-question/", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                                "X-CSRFToken": getCookie("csrftoken"),
                            },
                            body: JSON.stringify({
                                benchmark: "drive",
                                question: row.question
                            }),
                        });
                        const resData = await response.json();
                        if (!response.ok) throw new Error(resData.detail || "Échec de la suppression");
                        await refreshDashboard();
                    } catch (err) {
                        alert(err.message);
                    } finally {
                        deleteBtn.disabled = false;
                    }
                }
            });
            actionCell.appendChild(deleteBtn);

            tr.append(
                makeCell(row.question || "-"),
                makeCell(row.expected_service || "-"),
                makeCell(row.top1_service || "-"),
                statusCell,
                makeCell(percent(row.precision_at_k)),
                makeCell(row.latency_ms != null ? `${Math.round(row.latency_ms)} ms` : "-"),
                makeCell(row.top1_source || "-"),
                actionCell
            );
            body.appendChild(tr);
        });
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
            actionCell.style.cssText = "display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;";

            const downloadBtn = document.createElement("button");
            downloadBtn.className = "uca-btn uca-btn-secondary";
            downloadBtn.type = "button";
            downloadBtn.textContent = "Télécharger";
            downloadBtn.title = "Télécharger ce document";
            downloadBtn.addEventListener("click", () => downloadDriveDocument(doc.name || ""));
            actionCell.appendChild(downloadBtn);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "uca-btn uca-btn-danger";
            deleteBtn.type = "button";
            deleteBtn.textContent = "Supprimer";
            deleteBtn.title = "Supprimer définitivement ce document du corpus";
            deleteBtn.addEventListener("click", () => deleteDriveDocument(doc.name || ""));
            actionCell.appendChild(deleteBtn);
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

async function downloadDriveDocument(filename) {
    if (!filename) return;
    try {
        const encodedName = encodeURIComponent(filename);
        const response = await fetch(`/api/drive-documents/${encodedName}/download/`, {
            method: "GET",
        });
        if (!response.ok) {
            const payload = await response.json();
            throw new Error(payload.detail || "Téléchargement impossible.");
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        setActionStatus(`${filename} téléchargé avec succès.`);
    } catch (error) {
        setActionStatus(error.message || "Erreur pendant le téléchargement.", true);
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

async function rebuildDrive(clearCache = false) {
    if (clearCache) {
        if (!window.confirm("Êtes-vous sûr de vouloir réinitialiser le cache et recalculer tous les documents à zéro ? Cela peut prendre plusieurs minutes.")) {
            return;
        }
        setActionStatus("Rebuild à zéro en cours : réinitialisation du cache + traitement complet...");
    } else {
        setActionStatus("Rebuild drive en cours : processing + indexation publiee...");
    }
    renderProcessingSummary(null);
    try {
        const response = await fetch("/api/drive-rebuild/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({ clear_cache: clearCache }),
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

const QUALITY_AUDIT_TASKS = new Set(["data_audit", "raw_quality"]);

function appendAuditLog(message, isError = false, consoleId = "audit-console") {
    const consoleBox = byId(consoleId);
    if (!consoleBox) return;
    const line = document.createElement("div");
    line.className = isError ? "audit-log-line is-error" : "audit-log-line";
    line.textContent = `[${new Date().toLocaleTimeString("fr-FR")}] ${message}`;
    if (consoleBox.firstChild && consoleBox.children.length === 0) {
        clearNode(consoleBox);
    } else if (!consoleBox.querySelector(".audit-log-line")) {
        clearNode(consoleBox);
    }
    consoleBox.appendChild(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

async function runAuditTask(task, button) {
    if (!task || !button) return;
    // Route to the closest audit console in the same article/section, fallback to quality-audit-console
    const closestConsole = button.closest("article")?.querySelector(".audit-console");
    const consoleId = closestConsole
        ? closestConsole.id || "quality-audit-console"
        : (QUALITY_AUDIT_TASKS.has(task) ? "quality-audit-console" : "audit-console");
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "En cours...";
    appendAuditLog(`Lancement ${originalText}`, false, consoleId);
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
        appendAuditLog(`${payload.label || originalText} terminé en ${payload.elapsed_s || 0}s`, false, consoleId);
        await refreshDashboard();
    } catch (error) {
        appendAuditLog(`${originalText}: ${error.message || "erreur inconnue"}`, true, consoleId);
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

let convCurrentPage = 1;
let convStatusFilter = "active";
let convSearchQuery = "";

async function manageConversation(id, action) {
    if (action === "delete" && !confirm("Voulez-vous vraiment supprimer définitivement cette conversation ?")) {
        return;
    }
    try {
        const response = await fetch(`/api/conversations/${id}/manage/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken")
            },
            body: JSON.stringify({ action })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Opération échouée");
        await loadConversationAudit();
    } catch (err) {
        alert(err.message);
    }
}

async function openConversationDrawer(id) {
    const drawer = byId("conv-detail-drawer");
    const titleEl = byId("drawer-conv-title");
    const subtitleEl = byId("drawer-conv-subtitle");
    const messagesEl = byId("conv-drawer-messages");
    const archiveBtn = byId("drawer-archive-btn");
    if (!drawer || !messagesEl) return;

    // Reset drawer state
    clearNode(messagesEl);
    if (titleEl) titleEl.textContent = "Chargement...";
    if (subtitleEl) subtitleEl.textContent = "-";
    if (archiveBtn) {
        archiveBtn.style.display = "none";
        archiveBtn.onclick = null;
    }

    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");

    try {
        const response = await fetch(`/api/conversations/${id}/manage/`);
        if (!response.ok) throw new Error("Impossible de charger les messages.");
        const data = await response.json();

        if (titleEl) titleEl.textContent = data.title || "Conversation";
        if (subtitleEl) subtitleEl.textContent = `Utilisateur : ${data.user} • ${data.is_archived ? "Archivée" : "Active"}`;

        if (archiveBtn) {
            archiveBtn.style.display = "inline-block";
            archiveBtn.textContent = data.is_archived ? "Restaurer" : "Archiver";
            archiveBtn.onclick = async () => {
                const action = data.is_archived ? "restore" : "archive";
                await manageConversation(id, action);
                await openConversationDrawer(id); // Reload drawer to reflect status
            };
        }

        if (!data.messages || data.messages.length === 0) {
            messagesEl.innerHTML = '<div style="text-align: center; color: var(--uca-muted); margin-top: 40px; font-size: 13px;">Aucun message dans cette conversation.</div>';
            return;
        }

        data.messages.forEach(msg => {
            const chatMsg = document.createElement("div");
            chatMsg.className = `chat-msg ${msg.role}`;

            const bubble = document.createElement("div");
            bubble.className = "chat-msg-bubble";
            bubble.textContent = msg.content || "";

            const meta = document.createElement("div");
            meta.className = "chat-msg-meta";
            const dateStr = msg.created_at ? new Date(msg.created_at).toLocaleString("fr-FR", { hour: '2-digit', minute: '2-digit' }) : "";
            meta.textContent = `${msg.role === "user" ? "Étudiant" : "Assistant RAG"} • ${dateStr}`;

            if (msg.role === "assistant" && msg.sources && msg.sources.length > 0) {
                const sourcesList = document.createElement("div");
                sourcesList.style.marginTop = "8px";
                sourcesList.style.paddingTop = "6px";
                sourcesList.style.borderTop = "1px solid var(--uca-border)";
                sourcesList.style.fontSize = "11px";
                sourcesList.style.color = "var(--uca-blue)";
                
                const sourcesTitle = document.createElement("strong");
                sourcesTitle.textContent = "Sources utilisées :";
                sourcesList.appendChild(sourcesTitle);

                msg.sources.forEach(src => {
                    const srcLink = document.createElement("div");
                    srcLink.style.marginTop = "2px";
                    srcLink.textContent = `• ${src.name || src.path || "Source"}`;
                    sourcesList.appendChild(srcLink);
                });
                bubble.appendChild(sourcesList);
            }

            if (msg.role === "assistant" && msg.feedback) {
                const fbDiv = document.createElement("div");
                fbDiv.className = "msg-feedback-display";
                fbDiv.style.marginTop = "8px";
                fbDiv.style.padding = "6px 10px";
                fbDiv.style.borderRadius = "var(--uca-radius)";
                fbDiv.style.fontSize = "11px";
                fbDiv.style.background = msg.feedback.rating === "up" ? "#e6f4ea" : "#fce8e6";
                fbDiv.style.color = msg.feedback.rating === "up" ? "#137333" : "#c5221f";
                fbDiv.style.border = `1px solid ${msg.feedback.rating === "up" ? "#ceead6" : "#fad2cf"}`;
                
                const fbText = document.createElement("span");
                fbText.innerHTML = `<strong>Avis étudiant :</strong> ${msg.feedback.rating === "up" ? "👍 Utile" : "👎 Pas utile"}`;
                fbDiv.appendChild(fbText);

                if (msg.feedback.comment) {
                    const fbComment = document.createElement("div");
                    fbComment.style.marginTop = "4px";
                    fbComment.style.fontStyle = "italic";
                    fbComment.textContent = `"${msg.feedback.comment}"`;
                    fbDiv.appendChild(fbComment);
                }
                bubble.appendChild(fbDiv);
            }

            chatMsg.append(bubble, meta);
            messagesEl.appendChild(chatMsg);
        });
        messagesEl.scrollTop = messagesEl.scrollHeight;

    } catch (err) {
        messagesEl.innerHTML = `<div style="text-align: center; color: var(--uca-danger); margin-top: 40px; font-size: 13px;">Erreur : ${err.message}</div>`;
    }
}

function closeConversationDrawer() {
    const drawer = byId("conv-detail-drawer");
    if (drawer) {
        drawer.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
    }
}

async function loadConversationAudit() {
    const body = byId("conversation-audit-table-body");
    const tableInfo = byId("conv-table-info");
    const paginationEl = byId("conv-pagination");
    const prevBtn = byId("conv-prev-btn");
    const nextBtn = byId("conv-next-btn");
    const pageInfo = byId("conv-page-info");
    const perpageSelect = byId("conv-perpage-select");
    const feedbackOnlyCheckbox = byId("conv-feedback-only");
    if (!body) return;

    const perPage = perpageSelect ? parseInt(perpageSelect.value) || 25 : 25;
    const feedbackOnly = feedbackOnlyCheckbox && feedbackOnlyCheckbox.checked ? "1" : "";

    try {
        const queryParams = new URLSearchParams({
            status: convStatusFilter,
            search: convSearchQuery,
            page: convCurrentPage,
            per_page: perPage,
            has_feedback: feedbackOnly
        });
        const response = await fetch(`/api/admin-conversations/?${queryParams.toString()}`);
        if (response.status === 403) {
            body.replaceChildren(makeEmptyRow(8, "Accès refusé."));
            return;
        }
        if (!response.ok) {
            throw new Error("Chargement impossible.");
        }
        const payload = await response.json();
        
        const summary = payload.summary || {};
        setText("conv-active", String(summary.active_conversations || 0));
        setText("conv-archived", String(summary.archived_conversations || 0));
        setText("conv-messages", String(summary.total_messages || 0));
        setText("conv-answers", String(summary.assistant_answers || 0));
        setText("conv-with-sources", String(summary.answers_with_sources || 0));
        setText("conv-source-coverage", percent(summary.source_coverage));

        const pag = payload.pagination || {};
        if (tableInfo) {
            tableInfo.textContent = `${pag.total || 0} conversation(s) trouvée(s)`;
        }
        if (paginationEl && pag.total_pages > 1) {
            paginationEl.style.display = "flex";
            if (pageInfo) pageInfo.textContent = `Page ${pag.page} sur ${pag.total_pages}`;
            if (prevBtn) prevBtn.disabled = pag.page <= 1;
            if (nextBtn) nextBtn.disabled = pag.page >= pag.total_pages;
        } else if (paginationEl) {
            paginationEl.style.display = "none";
        }

        clearNode(body);
        const conversations = payload.recent || [];
        if (!conversations.length) {
            body.appendChild(makeEmptyRow(8, "Aucune conversation correspondante."));
            return;
        }
        conversations.forEach((item) => {
            const row = document.createElement("tr");
            row.style.cursor = "pointer";
            row.addEventListener("mouseover", () => {
                row.style.background = "var(--uca-surface-muted)";
            });
            row.addEventListener("mouseout", () => {
                row.style.background = "";
            });
            row.addEventListener("click", () => openConversationDrawer(item.id));

            const assistantCount = item.assistant_count || 0;
            const sourceCount = item.source_answer_count || 0;
            const sourceRate = assistantCount > 0 ? sourceCount / assistantCount : 0;

            const userCell = document.createElement("td");
            userCell.textContent = item.user || "-";
            if (item.feedback_count > 0) {
                const fbBadge = document.createElement("span");
                fbBadge.textContent = "👍 Avis";
                fbBadge.style.marginLeft = "8px";
                fbBadge.style.fontSize = "10px";
                fbBadge.style.fontWeight = "600";
                fbBadge.style.padding = "2px 6px";
                fbBadge.style.borderRadius = "10px";
                fbBadge.style.background = "#e6f4ea";
                fbBadge.style.color = "#137333";
                fbBadge.style.border = "1px solid #ceead6";
                userCell.appendChild(fbBadge);
            }

            const statusPill = makePill(
                item.is_archived ? "Archivée" : "Active",
                item.is_archived ? "neutral" : "ok"
            );
            const statusCell = document.createElement("td");
            statusCell.appendChild(statusPill);

            const actionsCell = document.createElement("td");
            actionsCell.style.display = "flex";
            actionsCell.style.justifyContent = "center";

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "uca-btn uca-btn-danger-outline";
            deleteBtn.textContent = "Supprimer";
            deleteBtn.addEventListener("click", (e) => {
                e.stopPropagation(); // Avoid triggering openConversationDrawer on row click
                manageConversation(item.id, "delete");
            });

            actionsCell.append(deleteBtn);

            row.append(
                userCell,
                makeCell(item.title || "-"),
                makeCell(String(item.message_count || 0)),
                makeCell(String(assistantCount)),
                makeCell(percent(sourceRate)),
                makeCell(formatDate(item.last_message_at)),
                statusCell,
                actionsCell
            );
            body.appendChild(row);
        });
    } catch (error) {
        clearNode(body);
        body.appendChild(makeEmptyRow(8, "Erreur de chargement des conversations."));
    }
}

async function testBenchmarkQuestion() {
    const questionInput = byId("new-question");
    const expectedServiceInput = byId("new-expected-service");
    const keywordsInput = byId("new-keywords");
    const testResultBox = byId("test-result-box");
    const testResultTbody = byId("test-result-tbody");
    const testBtn = byId("test-question-btn");

    if (!questionInput || !questionInput.value.trim()) {
        alert("Veuillez saisir une question.");
        return;
    }
    
    testBtn.disabled = true;
    testBtn.textContent = "Test en cours...";
    try {
        const response = await fetch("/api/benchmark/test-question/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({
                question: questionInput.value.trim(),
                expected_service: expectedServiceInput ? expectedServiceInput.value.trim() : "",
                keywords: keywordsInput ? keywordsInput.value.trim() : "",
            }),
        });
        const row = await response.json();
        if (!response.ok) throw new Error(row.detail || "Échec du test en direct");
        
        testResultTbody.replaceChildren();
        const tr = document.createElement("tr");
        const isMatch = Number(row.service_top1_match) === 1;
        const statusPill = makePill(isMatch ? "Correct" : "Erreur", isMatch ? "ok" : "bad");
        const statusCell = document.createElement("td");
        statusCell.appendChild(statusPill);
        
        tr.append(
            makeCell(row.question || "-"),
            makeCell(row.expected_service || "-"),
            makeCell(row.top1_service || "-"),
            statusCell,
            makeCell(percent(row.precision_at_k)),
            makeCell(row.latency_ms != null ? `${Math.round(row.latency_ms)} ms` : "-"),
            makeCell(row.top1_source || "-")
        );
        testResultTbody.appendChild(tr);
        testResultBox.style.display = "block";
    } catch (err) {
        alert(err.message);
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = "Tester en direct";
    }
}

async function addBenchmarkQuestion(e) {
    if (e) e.preventDefault();
    const questionInput = byId("new-question");
    const expectedServiceInput = byId("new-expected-service");
    const keywordsInput = byId("new-keywords");
    const addBtn = byId("add-question-btn");
    
    if (!questionInput || !questionInput.value.trim()) {
        alert("Veuillez saisir une question.");
        return;
    }
    if (!expectedServiceInput || !expectedServiceInput.value.trim()) {
        alert("Veuillez spécifier le service attendu.");
        return;
    }
    
    addBtn.disabled = true;
    addBtn.textContent = "Ajout en cours...";
    try {
        const response = await fetch("/api/benchmark/add-question/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({
                benchmark: "drive",
                question: questionInput.value.trim(),
                expected_service: expectedServiceInput.value.trim(),
                keywords: keywordsInput ? keywordsInput.value.trim() : "",
            }),
        });
        const resData = await response.json();
        if (!response.ok) throw new Error(resData.detail || "Échec de l'ajout");
        
        alert("Question ajoutée avec succès au benchmark !");
        questionInput.value = "";
        expectedServiceInput.value = "";
        if (keywordsInput) keywordsInput.value = "";
        
        const testResultBox = byId("test-result-box");
        if (testResultBox) testResultBox.style.display = "none";
        
        await refreshDashboard();
    } catch (err) {
        alert(err.message);
    } finally {
        addBtn.disabled = false;
        addBtn.textContent = "Ajouter au benchmark";
    }
}

function bindConversationEvents() {
    const searchInput = byId("conv-search-input");
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener("input", () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                convSearchQuery = searchInput.value;
                convCurrentPage = 1;
                loadConversationAudit();
            }, 300);
        });
    }

    const tabsContainer = byId("conv-filter-tabs");
    if (tabsContainer) {
        tabsContainer.querySelectorAll(".conv-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                tabsContainer.querySelectorAll(".conv-tab").forEach(t => {
                    t.classList.remove("active");
                    t.setAttribute("aria-selected", "false");
                });
                tab.classList.add("active");
                tab.setAttribute("aria-selected", "true");
                convStatusFilter = tab.dataset.status || "active";
                convCurrentPage = 1;
                loadConversationAudit();
            });
        });
    }

    bindAction("conv-refresh-btn", () => {
        loadConversationAudit();
    });

    bindAction("conv-prev-btn", () => {
        if (convCurrentPage > 1) {
            convCurrentPage--;
            loadConversationAudit();
        }
    });

    bindAction("conv-next-btn", () => {
        convCurrentPage++;
        loadConversationAudit();
    });

    const perpageSelect = byId("conv-perpage-select");
    if (perpageSelect) {
        perpageSelect.addEventListener("change", () => {
            convCurrentPage = 1;
            loadConversationAudit();
        });
    }

    const feedbackOnlyCheckbox = byId("conv-feedback-only");
    if (feedbackOnlyCheckbox) {
        feedbackOnlyCheckbox.addEventListener("change", () => {
            convCurrentPage = 1;
            loadConversationAudit();
        });
    }

    bindAction("conv-drawer-close", closeConversationDrawer);
    bindAction("conv-drawer-backdrop", closeConversationDrawer);
}

bindAction("upload-drive-btn", uploadDriveDocument);
bindAction("rebuild-drive-btn", () => rebuildDrive(false));
bindAction("rebuild-zero-btn", () => rebuildDrive(true));
bindAction("rebuild-drive-btn-bottom", () => rebuildDrive(false));
bindAction("rebuild-zero-btn-bottom", () => rebuildDrive(true));
bindAction("evaluate-drive-btn", evaluateDrive);
bindAction("evaluate-drive-btn-bottom", evaluateDrive);
bindAction("test-question-btn", testBenchmarkQuestion);

const addQuestionForm = byId("add-test-question-form");
if (addQuestionForm) {
    addQuestionForm.addEventListener("submit", addBenchmarkQuestion);
}

bindAction("clear-maint-console-btn", () => {
    const consoleBox = byId("maint-audit-console");
    if (consoleBox) {
        clearNode(consoleBox);
        const line = document.createElement("div");
        line.className = "audit-log-line";
        line.textContent = `[${new Date().toLocaleTimeString("fr-FR")}] Console vide.`;
        consoleBox.appendChild(line);
    }
});

async function loadHealthStatus() {
    const globalBox = byId("health-global-box");
    const globalDot = byId("health-global-dot");
    const globalTitle = byId("health-global-title");
    
    const dbPill = byId("health-status-db");
    const vectorPill = byId("health-status-vector");
    const llmPill = byId("health-status-llm");
    const refreshBtn = byId("health-modal-refresh");

    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.textContent = "Vérification...";
    }

    try {
        const response = await fetch("/api/health/ready/");
        const payload = await response.json();
        
        const isReady = Boolean(payload.ready);
        const checks = payload.checks || {};

        // Update DB
        if (dbPill) {
            dbPill.textContent = checks.database_ready ? "Opérationnel" : "Hors-ligne";
            dbPill.className = `status-pill ${checks.database_ready ? "status-ok" : "status-bad"}`;
        }
        // Update Vector Index
        if (vectorPill) {
            vectorPill.textContent = checks.vector_store_ready ? "Opérationnel" : "Non chargé";
            vectorPill.className = `status-pill ${checks.vector_store_ready ? "status-ok" : "status-bad"}`;
        }
        // Update LLM
        if (llmPill) {
            llmPill.textContent = checks.llm_ready ? "Disponible" : "Indisponible";
            llmPill.className = `status-pill ${checks.llm_ready ? "status-ok" : "status-bad"}`;
        }

        // Global Alert styling
        if (globalBox && globalDot && globalTitle) {
            if (isReady) {
                globalBox.style.backgroundColor = "#e6f4ea";
                globalBox.style.borderColor = "#ceead6";
                globalDot.style.backgroundColor = "#137333";
                globalTitle.textContent = "Tous les services sont opérationnels (Prêt).";
                globalTitle.style.color = "#137333";
            } else {
                globalBox.style.backgroundColor = "#fce8e6";
                globalBox.style.borderColor = "#fad2cf";
                globalDot.style.backgroundColor = "#c5221f";
                globalTitle.textContent = "Certains services nécessitent une attention.";
                globalTitle.style.color = "#c5221f";
            }
        }
    } catch (err) {
        if (globalTitle) globalTitle.textContent = "Erreur de connexion au healthcheck.";
        if (dbPill) dbPill.className = "status-pill status-bad";
        if (vectorPill) vectorPill.className = "status-pill status-bad";
        if (llmPill) llmPill.className = "status-pill status-bad";
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.textContent = "Re-tester";
        }
    }
}

function openHealthModal(e) {
    if (e) e.preventDefault();
    const modal = byId("health-modal");
    if (modal) {
        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
        loadHealthStatus();
    }
}

function closeHealthModal() {
    const modal = byId("health-modal");
    if (modal) {
        modal.classList.remove("open");
        modal.setAttribute("aria-hidden", "true");
    }
}

bindAction("sidebar-health-link", openHealthModal);
bindAction("health-modal-close", closeHealthModal);
bindAction("health-modal-close-btn", closeHealthModal);
bindAction("health-modal-backdrop", closeHealthModal);
bindAction("health-modal-refresh", loadHealthStatus);

bindConversationEvents();
bindAuditActions();
