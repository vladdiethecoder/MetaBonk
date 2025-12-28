import { test, expect } from "@playwright/test";

async function expectNotBlank(page: any) {
  await page.waitForTimeout(250);
  const bodyText = await page.locator("body").innerText();
  expect(bodyText.trim().length).toBeGreaterThan(20);
}

const routes = [
  { path: "/", text: "Lobby" },
  { path: "/neural", text: "Agent Thoughts" },
  { path: "/lab", text: "Laboratory" },
  { path: "/lab/runs", text: "Runs" },
  { path: "/lab/instances", text: "Instances" },
  { path: "/lab/build", text: "Build Lab" },
  { path: "/lab/clips", text: "Clips" },
  { path: "/codex", text: "Codex" },
  { path: "/codex/skills", text: "Skill Spy" },
];

test.describe("UI smoke", () => {
  for (const route of routes) {
    test(`renders ${route.path}`, async ({ page }) => {
      await page.goto(route.path);
      await expect(page.getByRole("heading", { name: route.text, exact: false }).first()).toBeVisible();
    });
  }

  test("broadcast route renders", async ({ page }) => {
    await page.goto("/broadcast");
    await expectNotBlank(page);
  });
});
