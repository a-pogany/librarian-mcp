import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

function parseToolResult(result) {
  if (!result) {
    return { error: "Empty MCP response" };
  }

  if (Array.isArray(result.content) && result.content.length > 0) {
    const first = result.content[0];
    if (first.type === "json" && first.json) {
      return first.json;
    }
    if (first.type === "text" && typeof first.text === "string") {
      try {
        return JSON.parse(first.text);
      } catch {
        return { rawText: first.text };
      }
    }
  }

  return result;
}

export class MCPClientManager {
  constructor({ url, logger }) {
    this.url = url;
    this.logger = logger;
    this.client = null;
    this.transport = null;
    this.connected = false;
    this.lastError = null;
    this.lastConnectedAt = null;
  }

  async connect() {
    if (this.connected && this.client) {
      return;
    }

    this.transport = new SSEClientTransport(new URL(this.url));
    this.client = new Client(
      { name: "librarian-agent-layer", version: "0.1.0" },
      { capabilities: {} }
    );

    try {
      await this.client.connect(this.transport);
      await this.client.listTools();
      this.connected = true;
      this.lastConnectedAt = new Date().toISOString();
      this.lastError = null;
    } catch (error) {
      this.connected = false;
      this.lastError = error instanceof Error ? error.message : String(error);
      throw error;
    }
  }

  async callTool(name, args) {
    if (!this.connected) {
      await this.connect();
    }

    const result = await this.client.callTool({
      name,
      arguments: args
    });

    return parseToolResult(result);
  }

  getStatus() {
    return {
      connected: this.connected,
      url: this.url,
      lastError: this.lastError,
      lastConnectedAt: this.lastConnectedAt
    };
  }
}
