import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: process.env.METABONK_UI_BASE_URL ?? "http://localhost:5173",
    viewport: { width: 1280, height: 720 },
  },
});
