import { test, expect } from "@playwright/test";

const API_BASE = "http://127.0.0.1:4010";

function mockStatus(route) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      mcp: {
        connected: true,
        url: "http://127.0.0.1:3001/mcp/sse",
        lastError: null,
        lastConnectedAt: new Date().toISOString()
      },
      llm: { enabled: false, provider: "ollama" }
    })
  });
}

function mockSearch(route) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      query: "sync status",
      results: [
        {
          id: "docs/guide.md",
          file_path: "docs/guide.md",
          file_name: "guide.md",
          snippet: "Sync status is visible in the dashboard...",
          relevance_score: 0.92
        },
        {
          id: "docs/runbook.md",
          file_path: "docs/runbook.md",
          file_name: "runbook.md",
          snippet: "Check the status page for sync issues...",
          relevance_score: 0.81
        }
      ],
      total: 2,
      search_mode: "auto"
    })
  });
}

function mockDocument(route) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      file_path: "docs/guide.md",
      file_name: "guide.md",
      last_modified: "2024-01-01T00:00:00",
      content: "Full document content for sync status."
    })
  });
}

test.describe("Librarian UI", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(`${API_BASE}/api/status`, mockStatus);
    await page.route(`${API_BASE}/api/search`, mockSearch);
    await page.route(`${API_BASE}/api/document`, mockDocument);
  });

  test("loads the main layout", async ({ page }) => {
    await page.goto(`/?api=${API_BASE}`);
    await expect(page.getByText("Librarian Search")).toBeVisible();
    await expect(page.getByPlaceholder("Search documentation or email threads...")).toBeVisible();
    await expect(page.getByText("Findings")).toBeVisible();
    await expect(page.getByText("Detail")).toBeVisible();
  });

  test("shows MCP connection status", async ({ page }) => {
    await page.goto(`/?api=${API_BASE}`);
    await expect(page.getByText("Connected")).toBeVisible();
  });

  test("returns search results and shows details", async ({ page }) => {
    await page.goto(`/?api=${API_BASE}`);
    await page.getByPlaceholder("Search documentation or email threads...").fill("sync status");
    await page.getByRole("button", { name: "Search" }).click();

    await expect(page.locator(".result-title", { hasText: "guide.md" })).toBeVisible();
    await expect(page.locator(".result-title", { hasText: "runbook.md" })).toBeVisible();

    await page.locator(".result-item", { hasText: "guide.md" }).first().click();
    await expect(page.getByText("Full document content for sync status.")).toBeVisible();
  });
});
