import { test, expect } from "@playwright/test";

async function expectNotBlank(page: any) {
  await page.waitForTimeout(250);
  const bodyText = await page.locator("body").innerText();
  expect(bodyText.trim().length).toBeGreaterThan(20);
}

const routes = [
  { path: "/", text: "Neuro-Synaptic Interface" },
  { path: "/overview", text: "Cluster Health" },
  { path: "/runs", text: "Run Constellation" },
  { path: "/instances", text: "Instance Lattice" },
  { path: "/build", text: "Build Lab" },
  { path: "/skills", text: "Skill Spy" },
  { path: "/spy", text: "Swarm Surveillance" },
];

test.describe("UI smoke", () => {
  for (const route of routes) {
    test(`renders ${route.path}`, async ({ page }) => {
      await page.goto(route.path);
      await expect(page.getByText(route.text, { exact: false })).toBeVisible();
    });
  }

  test("broadcast route renders", async ({ page }) => {
    await page.goto("/broadcast");
    await expectNotBlank(page);
  });
});
