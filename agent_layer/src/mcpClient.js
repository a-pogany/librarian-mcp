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
    this.connecting = null;
  }

  async disconnect() {
    if (this.transport && typeof this.transport.close === "function") {
      try {
        await this.transport.close();
      } catch (error) {
        this.logger?.warn?.("Failed to close MCP transport", error);
      }
    }
    this.transport = null;
    this.client = null;
    this.connected = false;
    this.connecting = null;
  }

  async connect() {
    if (this.connecting) {
      await this.connecting;
      return;
    }

    if (this.connected && this.client) {
      try {
        await this.client.listTools();
        return;
      } catch (error) {
        this.logger?.warn?.("MCP session stale, reconnecting", error);
        await this.disconnect();
      }
    }

    this.connecting = (async () => {
      this.transport = new SSEClientTransport(new URL(this.url));
      this.transport.onerror = (error) => {
        this.lastError = error instanceof Error ? error.message : String(error);
        this.connected = false;
      };
      this.transport.onclose = () => {
        this.connected = false;
      };

      this.client = new Client(
        { name: "librarian-agent-layer", version: "0.1.0" },
        { capabilities: {} }
      );

      try {
        await this.client.connect(this.transport);
        // Wait for initialization to complete before listing tools
        await new Promise(resolve => setTimeout(resolve, 100));
        await this.client.listTools();
        this.connected = true;
        this.lastConnectedAt = new Date().toISOString();
        this.lastError = null;
      } catch (error) {
        await this.disconnect();
        this.connected = false;
        this.lastError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.connecting = null;
      }
    })();

    await this.connecting;
  }

  async callTool(name, args) {
    const attemptCall = async () => {
      if (!this.connected) {
        await this.connect();
      }

      const result = await this.client.callTool({
        name,
        arguments: args
      });

      return parseToolResult(result);
    };

    try {
      return await attemptCall();
    } catch (error) {
      await this.disconnect();
      await this.connect();
      return await attemptCall();
    }
  }

  async ping() {
    try {
      await this.connect();
      await this.client.listTools();
      return true;
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error);
      await this.disconnect();
      return false;
    }
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
