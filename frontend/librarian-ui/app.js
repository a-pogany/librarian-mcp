const params = new URLSearchParams(window.location.search);
const API_BASE = params.get("api") || "http://127.0.0.1:4010";

const searchInput = document.getElementById("searchInput");
const searchButton = document.getElementById("searchButton");
const resultsList = document.getElementById("resultsList");
const resultsCount = document.getElementById("resultsCount");
const loadMoreButton = document.getElementById("loadMoreButton");
const detailBody = document.getElementById("detailBody");
const detailMeta = document.getElementById("detailMeta");
const searchMeta = document.getElementById("searchMeta");
const statusDot = document.getElementById("statusDot");
const statusValue = document.getElementById("statusValue");
const statusRefresh = document.getElementById("statusRefresh");
const searchType = document.getElementById("searchType");
const searchMode = document.getElementById("searchMode");
const themeSelect = document.getElementById("themeSelect");

let currentResults = [];
let activeIndex = null;
let systemThemeMedia = window.matchMedia("(prefers-color-scheme: dark)");
let systemThemeListener = null;
let lastQuery = "";
let lastSearchType = "emails";
let lastTotal = 0;
let currentMaxResults = 10;

function formatScore(result) {
  const rawScore =
    result.relevance_score ??
    result.rerank_score ??
    result.similarity_score ??
    result.keyword_score ??
    result.semantic_score;

  if (!Number.isFinite(rawScore)) {
    return "";
  }

  const precision = rawScore < 1 ? 4 : 2;
  return `Score ${rawScore.toFixed(precision)}`;
}

function formatRecipientList(value, maxItems = 2) {
  if (!value) return "—";
  if (Array.isArray(value)) {
    if (value.length <= maxItems) {
      return value.join(", ");
    }
    const visible = value.slice(0, maxItems).join(", ");
    return `${visible} +${value.length - maxItems} more`;
  }
  return value;
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightMatches(text, query) {
  if (!text || !query) return text;
  const keywords = query.toLowerCase().split(/\s+/).filter(kw => kw.length > 2);
  if (keywords.length === 0) return text;
  let highlighted = text;
  keywords.forEach(kw => {
    const regex = new RegExp(`(${escapeRegex(kw)})`, "gi");
    highlighted = highlighted.replace(regex, "<mark>$1</mark>");
  });
  return highlighted;
}

function explainScore(result) {
  const score =
    result.relevance_score ??
    result.rerank_score ??
    result.similarity_score ??
    result.keyword_score ??
    result.semantic_score;

  if (!Number.isFinite(score)) {
    return { label: "—", className: "quality-unknown" };
  }

  const mode = result.search_mode || "unknown";

  // RRF (Reciprocal Rank Fusion) produces small scores: 1/(60+rank)
  // Rank 1 ≈ 0.0164, Rank 10 ≈ 0.0143, Rank 50 ≈ 0.0091
  if (mode === "hybrid_rrf") {
    if (score >= 0.015) {
      return { label: "Excellent", className: "quality-excellent" };
    } else if (score >= 0.013) {
      return { label: "Good", className: "quality-good" };
    } else if (score >= 0.010) {
      return { label: "Fair", className: "quality-fair" };
    } else {
      return { label: "Weak", className: "quality-weak" };
    }
  }

  // All other modes use 0-1 scale
  // Keyword search tends to have lower scores, so adjust thresholds
  if (mode === "keyword") {
    if (score >= 0.6) {
      return { label: "Excellent", className: "quality-excellent" };
    } else if (score >= 0.35) {
      return { label: "Good", className: "quality-good" };
    } else if (score >= 0.15) {
      return { label: "Fair", className: "quality-fair" };
    } else {
      return { label: "Weak", className: "quality-weak" };
    }
  }

  // Semantic/hybrid/rerank/hyde - standard 0-1 thresholds
  if (score >= 0.7) {
    return { label: "Excellent", className: "quality-excellent" };
  } else if (score >= 0.5) {
    return { label: "Good", className: "quality-good" };
  } else if (score >= 0.3) {
    return { label: "Fair", className: "quality-fair" };
  } else {
    return { label: "Weak", className: "quality-weak" };
  }
}

function isEmailDoc(data, fallbackPath) {
  if (data?.metadata?.doc_type === "email") return true;
  if (data?.file_type === ".eml") return true;
  if (fallbackPath && fallbackPath.toLowerCase().endsWith(".eml")) return true;
  return false;
}

function resolveSystemTheme() {
  return systemThemeMedia.matches ? "dark" : "light";
}

function applyTheme(mode) {
  const resolved = mode === "system" ? resolveSystemTheme() : mode;
  document.body.dataset.theme = resolved;
  document.body.dataset.themeMode = mode;
}

function loadTheme() {
  const stored = window.localStorage.getItem("librarian-theme") || "system";
  themeSelect.value = stored;
  applyTheme(stored);
}

function setStatus({ connected, lastError }) {
  statusDot.classList.remove("connected", "disconnected");
  if (connected) {
    statusDot.classList.add("connected");
    statusValue.textContent = "Connected";
  } else {
    statusDot.classList.add("disconnected");
    statusValue.textContent = lastError ? `Disconnected: ${lastError}` : "Disconnected";
  }
}

async function checkStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`);
    const data = await response.json();
    setStatus(data.mcp || { connected: false, lastError: "Unknown" });
  } catch (error) {
    setStatus({ connected: false, lastError: "Agent offline" });
  }
}

function renderEmptyState(message) {
  resultsList.innerHTML = "";
  resultsCount.textContent = "0";
  detailMeta.textContent = message;
  detailBody.innerHTML = "";
  const wrapper = document.createElement("div");
  wrapper.className = "empty-state";
  const title = document.createElement("h3");
  title.textContent = message;
  const note = document.createElement("p");
  note.textContent = "Try a different query or check that the MCP server is running.";
  wrapper.appendChild(title);
  wrapper.appendChild(note);
  detailBody.appendChild(wrapper);
}

function renderResults(results) {
  currentResults = results;
  activeIndex = null;
  resultsList.innerHTML = "";
  const totalLabel = lastTotal && lastTotal > results.length ? ` / ${lastTotal}` : "";
  resultsCount.textContent = `${results.length}${totalLabel}`;
  loadMoreButton.disabled = !lastQuery || (lastTotal && results.length >= lastTotal) || currentMaxResults >= 50;

  if (results.length === 0) {
    renderEmptyState("No results");
    return;
  }

  results.forEach((result, index) => {
    const item = document.createElement("li");
    item.className = "result-item";
    item.addEventListener("click", () => selectResult(index));

    const titleText = result.title || result.subject || result.file_name || result.file_path || "Untitled";
    const title = document.createElement("div");
    title.className = "result-title";
    title.innerHTML = highlightMatches(titleText, lastQuery);

    const summary = document.createElement("div");
    summary.className = "result-summary";
    if (result.type === "email" && result.summary) {
      summary.innerHTML = highlightMatches(result.summary, lastQuery);
    }

    const meta = document.createElement("div");
    meta.className = "result-meta";
    const path = result.file_path || result.id || "";
    const score = formatScore(result);
    const typeLabel = result.type ? result.type.toUpperCase() : "";
    meta.textContent = [typeLabel, path, score].filter(Boolean).join(" • ");

    const quality = explainScore(result);
    const qualityBadge = document.createElement("span");
    qualityBadge.className = `quality-badge ${quality.className}`;
    qualityBadge.textContent = quality.label;

    const emailMeta = document.createElement("div");
    emailMeta.className = "result-meta";
    if (result.from || result.to || result.date) {
      const from = result.from ? `From: ${result.from}` : "";
      const to = result.to ? `To: ${formatRecipientList(result.to)}` : "";
      const date = result.date ? `Date: ${result.date}` : "";
      emailMeta.textContent = [from, to, date].filter(Boolean).join(" • ");
    }

    item.appendChild(title);
    if (summary.innerHTML) {
      item.appendChild(summary);
    }
    item.appendChild(meta);
    meta.appendChild(qualityBadge);
    if (emailMeta.textContent) {
      item.appendChild(emailMeta);
    }
    resultsList.appendChild(item);
  });
}

async function selectResult(index) {
  const result = currentResults[index];
  if (!result) {
    return;
  }

  const items = resultsList.querySelectorAll(".result-item");
  items.forEach((item, idx) => {
    item.classList.toggle("active", idx === index);
  });
  activeIndex = index;

  detailMeta.textContent = "Loading...";
  detailBody.innerHTML = "";

  const path = result.file_path || result.id;
  if (!path) {
    detailMeta.textContent = "Missing document path";
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/document`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    const data = await response.json();

    if (data.error) {
      detailMeta.textContent = data.error;
      detailBody.innerHTML = "";
      const errorLine = document.createElement("div");
      errorLine.className = "meta-line";
      errorLine.textContent = data.detail || "No detail";
      detailBody.appendChild(errorLine);
      return;
    }

    detailMeta.textContent = data.file_name || path;
    const metaLines = [];
    if (data.file_path) metaLines.push(`Path: ${data.file_path}`);
    if (data.last_modified) metaLines.push(`Updated: ${data.last_modified}`);

    detailBody.innerHTML = "";
    const metaLine = document.createElement("div");
    metaLine.className = "meta-line";
    metaLine.textContent = metaLines.join(" • ");
    detailBody.appendChild(metaLine);

    const emailMetaSource = data.metadata || data;
    if (emailMetaSource && isEmailDoc(data, path)) {
      const emailBlock = document.createElement("div");
      emailBlock.className = "detail-email";

      const header = document.createElement("div");
      header.className = "detail-email-header";
      header.textContent = emailMetaSource.subject || "No subject";
      emailBlock.appendChild(header);

      const rows = document.createElement("div");
      rows.className = "detail-email-rows";

      const rowData = [
        { label: "From", value: emailMetaSource.from || "—" },
        { label: "To", value: formatRecipientList(emailMetaSource.to) },
        { label: "Cc", value: emailMetaSource.cc ? formatRecipientList(emailMetaSource.cc) : null },
        { label: "Date", value: emailMetaSource.date || "—" }
      ];

      rowData.forEach((item) => {
        if (!item.value) return;
        const row = document.createElement("div");
        row.className = "detail-email-row";
        const label = document.createElement("div");
        label.className = "detail-email-label";
        label.textContent = item.label;
        const value = document.createElement("div");
        value.className = "detail-email-value";
        value.textContent = item.value;
        row.appendChild(label);
        row.appendChild(value);
        rows.appendChild(row);
      });

      if (emailMetaSource.attachment_count) {
        const row = document.createElement("div");
        row.className = "detail-email-row";
        const label = document.createElement("div");
        label.className = "detail-email-label";
        label.textContent = "Attachments";
        const value = document.createElement("div");
        value.className = "detail-email-value";
        value.textContent = `${emailMetaSource.attachment_count}`;
        row.appendChild(label);
        row.appendChild(value);
        rows.appendChild(row);
      }

      emailBlock.appendChild(rows);
      detailBody.appendChild(emailBlock);
    }

    const content = document.createElement("pre");
    content.textContent = data.content || "No content returned.";
    detailBody.appendChild(content);
  } catch (error) {
    detailMeta.textContent = "Failed to load document";
    detailBody.innerHTML = "";
    const errorLine = document.createElement("div");
    errorLine.className = "meta-line";
    errorLine.textContent = error.message;
    detailBody.appendChild(errorLine);
  }
}

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    return;
  }

  lastQuery = query;
  lastSearchType = searchType.value;
  currentMaxResults = 10;
  await executeSearch();
}

async function executeSearch() {
  searchMeta.textContent = "Searching...";
  searchButton.disabled = true;
  searchButton.textContent = "Searching...";
  searchButton.setAttribute("aria-busy", "true");

  try {
    const response = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: lastQuery,
        searchType: lastSearchType,
        maxResults: currentMaxResults,
        mode: searchMode ? searchMode.value : "auto"
      })
    });

    const data = await response.json();
    if (data.error) {
      searchMeta.textContent = data.error;
      renderEmptyState("Search failed");
      return;
    }

    const results = Array.isArray(data.results) ? data.results : [];
    lastTotal = Number.isInteger(data.total) ? data.total : results.length;
    renderResults(results);
    const mode = data.search_mode ? `Mode: ${data.search_mode}` : "";
    const total = data.total ? `Total: ${data.total}` : "";
    searchMeta.textContent = [mode, total].filter(Boolean).join(" • ") || "Search complete.";
  } catch (error) {
    searchMeta.textContent = "Search failed";
    renderEmptyState("Agent offline");
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Search";
    searchButton.removeAttribute("aria-busy");
  }
}

searchButton.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});
statusRefresh.addEventListener("click", checkStatus);
loadMoreButton.addEventListener("click", async () => {
  if (!lastQuery) return;
  currentMaxResults = Math.min(currentMaxResults + 10, 50);
  await executeSearch();
});
themeSelect.addEventListener("change", (event) => {
  const mode = event.target.value;
  window.localStorage.setItem("librarian-theme", mode);
  applyTheme(mode);
});

checkStatus();
setInterval(checkStatus, 6000);
loadTheme();

systemThemeListener = () => {
  if (themeSelect.value === "system") {
    applyTheme("system");
  }
};
systemThemeMedia.addEventListener("change", systemThemeListener);

// Help modal
const helpModal = document.getElementById("helpModal");
const searchModeHelp = document.getElementById("searchModeHelp");
const helpModalClose = document.getElementById("helpModalClose");

function openHelpModal() {
  helpModal.classList.add("visible");
}

function closeHelpModal() {
  helpModal.classList.remove("visible");
}

if (searchModeHelp) {
  searchModeHelp.addEventListener("click", openHelpModal);
}

if (helpModalClose) {
  helpModalClose.addEventListener("click", closeHelpModal);
}

if (helpModal) {
  helpModal.addEventListener("click", (event) => {
    if (event.target === helpModal) {
      closeHelpModal();
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && helpModal.classList.contains("visible")) {
    closeHelpModal();
  }
});
