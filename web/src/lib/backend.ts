import { Client } from "@gradio/client";
import {
  buildScenarioComparisonRows,
  type ScenarioComparisonRow,
} from "@/scenarioComparison";

// The two frozen Gradio backends. In production they are co-served under same-origin paths;
// Vite dev overrides these paths with its local proxies below.
const PUBLIC_ORIGIN = window.location.origin;
const D1 = import.meta.env.VITE_D1_URL || `${PUBLIC_ORIGIN}/g1`; // Narrative → Scenario
const D2 = import.meta.env.VITE_D2_URL || `${PUBLIC_ORIGIN}/g2`; // Scenario → Narrative

// Connect FRESH per call — do NOT memoize. The Gradio dropdowns are stateful (analogue_scope
// narrows after a run), so a reused session rejects a fixed input on the 2nd call. A new session
// resets choices. Per owner: every generation is in-flight; on failure we throw (no cache).
const app1 = () => Client.connect(D1);
const app2 = () => Client.connect(D2);

// ---------------------------------------------------------------------------------------------
// Demo 1 — Narrative → Scenario
// ---------------------------------------------------------------------------------------------
export type Series = { p10: number[]; p50: number[]; p90: number[]; mean: number[]; paths: number[][] };
export type FactorFan = { market: string; display: string; valueKind: string; startLevel: number; cond: Series; base: Series };
export type Analogue = { label: string; weight: number; cosine: number; pathCount: number };
export type LiveScenario = {
  windowIndex: number;
  factors: Record<string, FactorFan>; // keyed by market (all 15)
  comparisonRows: ScenarioComparisonRow[];
  analogues: Analogue[];
  ess: number;
  support: string;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function series(row: any): Series {
  const S0 = Number(row.start_level);
  const raw = row.value_kind === "raw_level";
  const lv = (v: number) => (raw ? v : S0 + v);
  const arr = (a: number[] | undefined) => [S0, ...(a ?? []).map(lv)];
  const paths: number[][] = (row.sample_paths ?? []).map((p: number[] | { values?: number[] }) => {
    const vals = Array.isArray(p) ? p : (p.values ?? []);
    return [S0, ...vals.map(lv)];
  });
  return { p10: arr(row.p10), p50: arr(row.p50), p90: arr(row.p90), mean: arr(row.mean), paths };
}

/** Narrative → multi-factor 30-day scenario. One call returns ALL factors (cond + start-only
 *  baseline) + the real retrieved analogues. Throws on any failure (no cached fallback). */
export async function generateScenario(story: string, windowIndex: number): Promise<LiveScenario> {
  const c = await app1();
  const r = await c.predict("/run_live_openai_prefix_for_app", ["SPX", "ALL", story, windowIndex]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const report = JSON.parse(String((r as any).data[7]));
  const g = report?.generation;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const condRows: any[] = g?.path_quantiles ?? [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const baseRows: any[] = g?.start_only_baseline?.path_quantiles ?? [];
  if (!condRows.length) throw new Error("backend returned no path_quantiles");
  const factors: Record<string, FactorFan> = {};
  for (const cr of condRows) {
    if (!Array.isArray(cr.p50)) continue;
    const br = baseRows.find((b) => b.market === cr.market);
    factors[cr.market] = { market: cr.market, display: cr.display_name || cr.market, valueKind: cr.value_kind, startLevel: Number(cr.start_level), cond: series(cr), base: br ? series(br) : series(cr) };
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sel: any[] = g?.posterior_ensemble?.selected_support ?? [];
  const analogues: Analogue[] = sel
    .map((s) => ({
      label: String(s.history_start_date ?? s.window_id ?? "").slice(0, 10),
      weight: Number(s.base_support_weight ?? 0),
      cosine: Number(s.memory_support_cosine ?? 0),
      pathCount: Number(s.component_sample_count ?? 0),
    }))
    .filter((a) => a.weight > 0 || a.cosine > 0 || a.pathCount > 0)
    .sort((a, b) => b.weight - a.weight);
  const sW = analogues.reduce((s, a) => s + a.weight, 0) || 1;
  const sW2 = analogues.reduce((s, a) => s + a.weight * a.weight, 0) || 1;
  const ess = (sW * sW) / sW2; // effective number of analogues
  const topCos = analogues.length ? Math.max(...analogues.map((a) => a.cosine)) : 0;
  const support = analogues.length ? `Top similarity ${topCos.toFixed(2)}` : "No historical precedents";
  return {
    windowIndex,
    factors,
    comparisonRows: buildScenarioComparisonRows(report),
    analogues,
    ess,
    support,
  };
}

/** 30-day market history (the conditioning context) for a start window, all factors. Real data,
 *  fetched live from a read-only endpoint. */
export async function fetchContext(windowIndex: number): Promise<Record<string, number[]>> {
  const c = await app1();
  const r = await c.predict("/start_window_context", [windowIndex]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return JSON.parse(String((r as any).data[0]));
}

/** Searchable [date, window-index] start-date list from the Demo 1 config (the day-0 picker). */
export async function fetchStartDates(): Promise<{ date: string; index: number }[]> {
  return (await findDropdownChoices(D1, "Starting market state")).map((c) => ({ date: String(c[0]), index: Number(c[1]) }));
}

// ---------------------------------------------------------------------------------------------
// Demo 2 — Scenario → Narrative
// ---------------------------------------------------------------------------------------------
export type FactorRow = { factor: string; start: string; end: string; delta: string; direction: string; magnitude: string; confidence: string };
export type NarrativePacket = { markdown: string; scenario: FactorRow[] };

/** Full searchable historical-date list (start dates) from the Demo 2 config. */
export async function fetchHistoricalDates(): Promise<string[]> {
  return (await findDropdownChoices(D2, "Select the historical period")).map((c) => String(c[0]));
}

/** The scenario's observed 30-day forward factor paths (for the movement sparklines), keyed by
 *  FACTOR (uppercase, matching the scenario table). Real data, fetched live. */
export async function fetchScenarioMovement(date: string): Promise<Record<string, number[]>> {
  const c = await app2();
  const r = await c.predict("/scenario_movement", [date]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return JSON.parse(String((r as any).data[0]));
}

/** Historical scenario → grounded narrative packet (live, ~2 min) on ONE session. Step 1
 *  (historical, fast) yields the per-factor scenario table via onScenario so the UI can show it
 *  immediately; Step 2 (Codex author, slow) returns the narrative. Throws on failure (no cache). */
export async function generateNarrative(date: string, onScenario?: (s: FactorRow[]) => void): Promise<NarrativePacket> {
  const c = await app2();
  const cfg = await fetch(`${D2}/config`).then((r) => r.json());
  let cards: string | null = null, outdir: string | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const comp of cfg.components as any[]) {
    const label = String(comp?.props?.label || "");
    if (label.includes("Reference cards")) cards = comp.props.value;
    if (label.toLowerCase().includes("output directory")) outdir = comp.props.value;
  }
  // Step 1 — read the scenario's factor moves (fast)
  const h = await c.predict("/historical_scenario", [date, cards]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const hd = (h as any).data;
  const scenario = parseFrame(hd[1]); // Numerical scenario DataFrame
  const sidecar = hd[4];
  onScenario?.(scenario);
  // Step 2 — LLM author writes the packet (slow)
  const g = await c.predict("/generate_narrative", [sidecar, cards, outdir]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return { markdown: String((g as any).data[1]), scenario };
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------
async function findDropdownChoices(base: string, labelMatch: string): Promise<[string, number | string][]> {
  const cfg = await fetch(`${base}/config`).then((r) => r.json());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const comp of cfg.components as any[]) {
    const label = String(comp?.props?.label || "");
    if (label.includes(labelMatch) && Array.isArray(comp?.props?.choices)) return comp.props.choices;
  }
  return [];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseFrame(frame: any): FactorRow[] {
  const headers: string[] = (frame?.headers ?? []).map((h: string) => h.toLowerCase());
  const rows: unknown[][] = frame?.data ?? [];
  const at = (row: unknown[], name: string) => { const i = headers.indexOf(name); return i >= 0 ? String(row[i]) : ""; };
  return rows.map((row) => ({ factor: at(row, "factor"), start: at(row, "start"), end: at(row, "end"), delta: at(row, "delta"), direction: at(row, "direction"), magnitude: at(row, "magnitude"), confidence: at(row, "confidence") }));
}
