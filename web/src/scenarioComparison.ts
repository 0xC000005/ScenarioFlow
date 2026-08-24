export const COMPARISON_MARKETS = [
  { key: "IV_SURFACE", label: "IV surface" },
  { key: "SPX", label: "SPX" },
  { key: "US2Y", label: "US2Y" },
  { key: "US10Y", label: "US10Y" },
  { key: "BBB_OAS", label: "BBB OAS" },
  { key: "AAA_OAS", label: "AAA OAS" },
  { key: "USDJPY", label: "USDJPY" },
  { key: "DXY", label: "DXY" },
  { key: "GOLD", label: "Gold" },
  { key: "CRUDE_OIL", label: "Crude oil" },
  { key: "VIX", label: "VIX" },
] as const;

export type DirectionLabel = "Up" | "Down" | "Mixed" | "n/a";
export type Tone = "green" | "red" | "amber" | "gray";

export const PAPER_TONE_COLORS: Record<Tone, string> = {
  green: "#16803A",
  red: "#C53030",
  gray: "#5F6368",
  amber: "#B7791F",
};

export type DistributionReadout = {
  direction: DirectionLabel;
  majorityShare: string;
  meanMove: string;
};

export type ScenarioComparisonRow = {
  marketKey: string;
  marketLabel: string;
  startOnly: DistributionReadout;
  narrative: DistributionReadout;
  comparison: string;
};

type RawTerminalRow = {
  market?: unknown;
  mean_terminal_delta?: unknown;
  p10?: unknown;
  p90?: unknown;
  terminal_mean_sigma?: unknown;
  terminal_probability_up?: unknown;
  terminal_probability_down?: unknown;
};

type Presentation = { arrow: string; tone: Tone };

const NORMAL_P10_P90_SPAN = 2.5631031310892007;

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function finite(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function rowsByMarket(value: unknown): Map<string, RawTerminalRow> {
  const result = new Map<string, RawTerminalRow>();
  if (!Array.isArray(value)) return result;
  for (const item of value) {
    const row = record(item) as RawTerminalRow;
    const market = typeof row.market === "string" ? row.market.trim() : "";
    if (market) result.set(market, row);
  }
  return result;
}

function terminalMean(row: RawTerminalRow | undefined): number | null {
  return finite(row?.mean_terminal_delta);
}

function terminalBandWidth(row: RawTerminalRow | undefined): number | null {
  const p10 = finite(row?.p10);
  const p90 = finite(row?.p90);
  return p10 === null || p90 === null ? null : Math.abs(p90 - p10);
}

function probabilityPair(row: RawTerminalRow | undefined): [number, number] | null {
  const up = finite(row?.terminal_probability_up);
  const down = finite(row?.terminal_probability_down);
  if (up === null || down === null) return null;
  return [Math.max(0, Math.min(1, up)), Math.max(0, Math.min(1, down))];
}

function meanDirection(row: RawTerminalRow | undefined): DirectionLabel {
  const mean = terminalMean(row);
  if (mean === null) return "n/a";
  const width = terminalBandWidth(row);
  const threshold = Math.max(0.05 * (width || Math.abs(mean) || 1), 1e-8);
  if (Math.abs(mean) <= threshold) return "Mixed";
  return mean > 0 ? "Up" : "Down";
}

function direction(row: RawTerminalRow | undefined): DirectionLabel {
  const probabilities = probabilityPair(row);
  if (!probabilities) return meanDirection(row);
  const [up, down] = probabilities;
  if (up >= 0.6 && up > down) return "Up";
  if (down >= 0.6 && down > up) return "Down";
  return "Mixed";
}

function percentLabel(value: number): string {
  const scaled = value * 100;
  const lower = Math.floor(scaled);
  const fraction = scaled - lower;
  const rounded = Math.abs(fraction - 0.5) <= 1e-10
    ? (lower % 2 === 0 ? lower : lower + 1)
    : Math.round(scaled);
  return `${rounded}%`;
}

function majorityShare(row: RawTerminalRow | undefined): string {
  const probabilities = probabilityPair(row);
  if (!probabilities) return "n/a";
  const [up, down] = probabilities;
  if (Math.abs(up - down) <= 0.005) return `${percentLabel(Math.max(up, down))} split`;
  return up >= down ? `${percentLabel(up)} up` : `${percentLabel(down)} down`;
}

function signedNumber(value: number, digits: number): string {
  const rounded = Number(value.toFixed(digits));
  const normalized = Object.is(rounded, -0) ? 0 : rounded;
  return `${normalized >= 0 ? "+" : ""}${normalized.toFixed(digits)}`;
}

function rawMove(market: string, mean: number): string {
  if (market === "IV_SURFACE" || market.startsWith("IV_")) {
    return `${signedNumber(mean * 100, 1)} vol pts`;
  }
  if (["US2Y", "US10Y", "BBB_OAS", "AAA_OAS"].includes(market)) {
    return `${signedNumber(mean * 100, 1)} bp`;
  }
  if (["SPX", "GOLD"].includes(market)) return `${signedNumber(mean, 0)} pts`;
  return `${signedNumber(mean, 1)} pts`;
}

function sigmaScore(row: RawTerminalRow | undefined): number | null {
  const mean = terminalMean(row);
  if (mean === null) return null;
  const explicit = finite(row?.terminal_mean_sigma);
  if (explicit !== null) return explicit;
  const width = terminalBandWidth(row);
  if (width === null || width <= 1e-12) return null;
  return mean / (width / NORMAL_P10_P90_SPAN);
}

function meanMove(row: RawTerminalRow | undefined, market: string): string {
  const mean = terminalMean(row);
  if (mean === null) return "n/a";
  const raw = rawMove(market, mean);
  const sigma = sigmaScore(row);
  return sigma === null ? raw : `${raw} / ${signedNumber(sigma, 1)}σ`;
}

function comparison(
  narrative: RawTerminalRow | undefined,
  baseline: RawTerminalRow | undefined,
): string {
  const narrativeMean = terminalMean(narrative);
  const baselineMean = terminalMean(baseline);
  if (narrativeMean === null || baselineMean === null) return "Baseline unavailable";
  const threshold = Math.max(
    0.1 * Math.max(terminalBandWidth(narrative) || 0, terminalBandWidth(baseline) || 0),
    1e-8,
  );
  const delta = narrativeMean - baselineMean;
  if (Math.abs(delta) <= threshold) return "Similar";
  const narrativeView = meanDirection(narrative);
  if (delta > 0) {
    if (narrativeView === "Down") return "Less down than baseline";
    if (narrativeView === "Up") return "More up than baseline";
    return "Higher than baseline";
  }
  if (narrativeView === "Down") return "More down than baseline";
  if (narrativeView === "Up") return "Less up than baseline";
  return "Lower than baseline";
}

function readout(row: RawTerminalRow | undefined, market: string): DistributionReadout {
  return {
    direction: direction(row),
    majorityShare: majorityShare(row),
    meanMove: meanMove(row, market),
  };
}

export function buildScenarioComparisonRows(reportValue: unknown): ScenarioComparisonRow[] {
  const generation = record(record(reportValue).generation);
  const narrativeRows = rowsByMarket(generation.terminal_delta_summary);
  const baselineRows = rowsByMarket(record(generation.start_only_baseline).terminal_delta_summary);
  return COMPARISON_MARKETS.flatMap(({ key, label }) => {
    const narrative = narrativeRows.get(key);
    if (!narrative) return [];
    const baseline = baselineRows.get(key);
    return [
      {
        marketKey: key,
        marketLabel: label,
        startOnly: readout(baseline, key),
        narrative: readout(narrative, key),
        comparison: comparison(narrative, baseline),
      },
    ];
  });
}

export function directionPresentation(value: DirectionLabel): Presentation {
  if (value === "Up") return { arrow: "↑", tone: "green" };
  if (value === "Down") return { arrow: "↓", tone: "red" };
  return { arrow: "", tone: "gray" };
}

export function comparisonPresentation(value: string): Presentation {
  if (["More up than baseline", "Higher than baseline"].includes(value)) {
    return { arrow: "↑", tone: "green" };
  }
  if (["More down than baseline", "Lower than baseline"].includes(value)) {
    return { arrow: "↓", tone: "red" };
  }
  if (value === "Less down than baseline") return { arrow: "↓", tone: "amber" };
  if (value === "Less up than baseline") return { arrow: "↑", tone: "amber" };
  if (["Similar", "Similar to baseline"].includes(value)) {
    return { arrow: "—", tone: "gray" };
  }
  return { arrow: "", tone: "gray" };
}
