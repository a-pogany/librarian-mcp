const path = require("path");
const { defineConfig } = require("@playwright/test");

const rootDir = path.resolve(__dirname, "..", "..");

module.exports = defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:4170",
    trace: "retain-on-failure"
  },
  webServer: {
    command: `python3 -m http.server 4170 --directory ${path.join(rootDir, "frontend", "librarian-ui")}`,
    url: "http://127.0.0.1:4170",
    reuseExistingServer: true,
    timeout: 10_000
  }
});
