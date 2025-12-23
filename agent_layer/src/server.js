import "dotenv/config";
import express from "express";
import cors from "cors";
import { MCPClientManager } from "./mcpClient.js";
import { rewriteQueryWithOpenAI } from "./llm/openai.js";
import { rewriteQueryWithOllama } from "./llm/ollama.js";

const MCP_SSE_URL = process.env.MCP_SSE_URL || "http://127.0.0.1:3001/mcp/sse";
const PORT = Number.parseInt(process.env.AGENT_PORT || "4010", 10);
const AGENT_USE_LLM = process.env.AGENT_USE_LLM === "true";
const LLM_PROVIDER = process.env.LLM_PROVIDER || "ollama";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const mcp = new MCPClientManager({
  url: MCP_SSE_URL,
  logger: console
});

async function rewriteQuery(query) {
  if (!AGENT_USE_LLM) {
    return { rewrittenQuery: query, skipped: "disabled" };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);

  try {
    if (LLM_PROVIDER === "openai") {
      return await rewriteQueryWithOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        model: process.env.OPENAI_MODEL || "gpt-4o-mini",
        baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com",
        query,
        signal: controller.signal
      });
    }

    return await rewriteQueryWithOllama({
      baseUrl: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
      model: process.env.OLLAMA_MODEL || "llama3.1",
      query,
      signal: controller.signal
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { rewrittenQuery: query, skipped: "llm_error", detail: message };
  } finally {
    clearTimeout(timeout);
  }
}

app.get("/api/status", async (_req, res) => {
  try {
    await mcp.connect();
  } catch {
    // Connection errors are surfaced in status response.
  }

  res.json({
    mcp: mcp.getStatus(),
    llm: {
      enabled: AGENT_USE_LLM,
      provider: LLM_PROVIDER
    }
  });
});

app.post("/api/search", async (req, res) => {
  const { query, searchType, mode, maxResults } = req.body || {};

  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing query" });
    return;
  }

  const normalizedSearchType = searchType === "emails" ? "emails" : "documentation";
  const toolName = normalizedSearchType === "emails" ? "search_emails" : "search_documentation";

  const rewrite = await rewriteQuery(query);
  const effectiveQuery = rewrite.rewrittenQuery || query;

  try {
    const response = await mcp.callTool(toolName, {
      query: effectiveQuery,
      max_results: Number.isInteger(maxResults) ? maxResults : 10,
      mode: mode || "auto"
    });

    res.json({
      ...response,
      search_type: normalizedSearchType,
      original_query: query,
      rewritten_query: effectiveQuery,
      rewrite_meta: rewrite
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(502).json({ error: "MCP search failed", detail: message });
  }
});

app.post("/api/document", async (req, res) => {
  const { path, section } = req.body || {};

  if (!path || typeof path !== "string") {
    res.status(400).json({ error: "Missing document path" });
    return;
  }

  try {
    const response = await mcp.callTool("get_document", {
      path,
      section: typeof section === "string" ? section : undefined
    });

    res.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(502).json({ error: "MCP get_document failed", detail: message });
  }
});

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`Agent layer listening on http://127.0.0.1:${PORT}`);
});
