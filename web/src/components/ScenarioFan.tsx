import { useEffect, useState, type ComponentType } from "react";
import type { Series } from "@/lib/backend";

const INK = "#1d1d1f", BLUE = "#0071e3", GREY = "#86868b", GRID = "#ececef", HAIR = "#d2d2d7", TEAL = "#00897B";
const PATH_COLORS = ["#2E7D32", "#EF6C00", "#6A1B9A", "#00838F", "#AD1457", "#5D4037"];
const sl = { shape: "spline" as const, smoothing: 0.6 };

export type FanFmt = (v: number) => string;

// Rich scenario fan: conditioning context (history) + forward cond band/median/mean/paths, with an
// optional start-only baseline overlay (band + dashed median) toggled on/off. Per-factor formatted.
export function ScenarioFan({
  cond, base, history, showBase, fmt, height = 380, unit = "",
}: {
  cond: Series;
  base?: Series;
  history?: number[]; // levels for days [-(n-1) .. 0], ending at day 0 = start_level
  showBase?: boolean;
  fmt?: FanFmt;
  height?: number;
  unit?: string;
}) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [Plot, setPlot] = useState<ComponentType<any> | null>(null);
  useEffect(() => {
    let ok = true;
    (async () => {
      const factory = (await import("react-plotly.js/factory")).default;
      // @ts-expect-error - plotly.js-dist-min ships no types
      const Plotly = (await import("plotly.js-dist-min")).default;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (ok) setPlot(() => factory(Plotly as any) as ComponentType<any>);
    })();
    return () => { ok = false; };
  }, []);
  if (!Plot) return <div style={{ height }} />;

  const n = cond.p50.length; // includes day 0
  const fx = Array.from({ length: n }, (_, i) => i);
  const f = fmt ?? ((v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2 }));
  const hov = (name: string) => `${name} %{y:.2f}${unit}<extra></extra>`;
  const hxStart = history && history.length ? -(history.length - 1) : 0;
  const hx = history && history.length ? history.map((_, i) => hxStart + i) : [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [];
  // start-only baseline band + dashed median (behind)
  if (showBase && base) {
    data.push(
      { x: fx, y: base.p90, mode: "lines", line: { width: 0, ...sl }, hoverinfo: "skip", showlegend: false },
      { x: fx, y: base.p10, mode: "lines", line: { width: 0, ...sl }, fill: "tonexty", fillcolor: "rgba(134,134,139,0.13)", hoverinfo: "skip", name: "start-only band" },
      { x: fx, y: base.p50, mode: "lines", line: { color: GREY, width: 1.8, dash: "dash", ...sl }, hovertemplate: hov("start-only"), name: "start-only median" },
    );
  }
  // conditioned band
  data.push(
    { x: fx, y: cond.p90, mode: "lines", line: { width: 0, ...sl }, hoverinfo: "skip", showlegend: false },
    { x: fx, y: cond.p10, mode: "lines", line: { width: 0, ...sl }, fill: "tonexty", fillcolor: "rgba(0,113,227,0.13)", hoverinfo: "skip", name: "P10–P90" },
  );
  // sample paths
  cond.paths.slice(0, 6).forEach((p, i) => data.push({ x: fx, y: p, mode: "lines", line: { color: PATH_COLORS[i % PATH_COLORS.length], width: 0.9, ...sl }, opacity: 0.5, hoverinfo: "skip", showlegend: false }));
  // conditioned mean (dotted) + median (solid)
  data.push(
    { x: fx, y: cond.mean, mode: "lines", line: { color: TEAL, width: 1.6, dash: "dot", ...sl }, hovertemplate: hov("mean"), name: "mean" },
    { x: fx, y: cond.p50, mode: "lines", line: { color: BLUE, width: 2.8, ...sl }, hovertemplate: hov("median"), name: "median" },
  );
  // conditioning context (history) up to day 0
  if (hx.length) data.push({ x: hx, y: history, mode: "lines", line: { color: INK, width: 2, ...sl }, hovertemplate: hov("history"), showlegend: false });

  const xmin = hx.length ? hxStart : 0;
  const layout = {
    autosize: true, height, margin: { l: 56, r: 14, t: 8, b: 28 }, paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "-apple-system, Inter, sans-serif", color: GREY, size: 12 },
    xaxis: { range: [xmin, n - 1], tickvals: [xmin, 0, 10, 20, 30].filter((v) => v >= xmin), ticktext: [xmin === 0 ? "Today" : `${xmin}d`, "Today", "+10d", "+20d", "+30d"].filter((_, i) => [xmin, 0, 10, 20, 30][i] >= xmin), showgrid: false, zeroline: false, linecolor: HAIR, fixedrange: true },
    yaxis: { showgrid: true, gridcolor: GRID, zeroline: false, tickformat: ",.0f", fixedrange: true, tickprefix: "", ticksuffix: unit },
    showlegend: false,
    shapes: [
      ...(hx.length ? [{ type: "rect", x0: xmin, x1: 0, y0: 0, y1: 1, yref: "paper", fillcolor: "rgba(0,0,0,0.035)", line: { width: 0 }, layer: "below" }] : []),
      { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { color: HAIR, width: 1 } },
    ],
    annotations: [
      ...(hx.length ? [{ x: xmin, y: 1, yref: "paper", xanchor: "left", yanchor: "top", text: "conditioning context", showarrow: false, font: { color: GREY, size: 11 } }] : []),
      { x: n - 1, y: 1, yref: "paper", xanchor: "right", yanchor: "top", text: showBase && base ? "—— narrative   – – start-only" : "—— narrative", showarrow: false, font: { color: GREY, size: 11 } },
    ],
    hovermode: "x unified",
  };
  void f;
  return <Plot data={data as never} layout={layout as never} config={{ displayModeBar: false, responsive: true } as never} style={{ width: "100%", height }} useResizeHandler />;
}
