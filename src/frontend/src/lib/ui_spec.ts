export type UiSpec = {
  version: string;
  layout: {
    resolution: { w: number; h: number };
    safe: { graphics: { x: number; y: number }; action: { x: number; y: number } };
    layoutScale: { broadcast: number; dense: number };
  };
  typography: { headingMin: number; labelMin: number; microMin: number; tickerMin: number };
  events: { schema: string[]; types: string[]; severity: string[] };
  icons: unknown;
};

export async function fetchUiSpec(): Promise<UiSpec> {
  const res = await fetch("/ui_spec.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`ui_spec.json load failed: ${res.status}`);
  return res.json();
}
