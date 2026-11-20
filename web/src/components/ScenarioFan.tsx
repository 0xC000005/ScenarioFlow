import { useEffect, useRef, useState, type ComponentType } from "react";
import type { Series } from "@/lib/backend";
import { VG, VG_FONT } from "@/lib/palette";

// Palette: red = + your words (conditioned), grey = market state only, black = history.
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
  // container width: narrow layouts get shorter labels, a smaller margin and fewer ticks
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(800);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cw = entries[0]?.contentRect.width;
      if (cw) setW(Math.round(cw));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [Plot]);
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
  const narrow = W < 560;
  const h = narrow ? Math.min(height, 320) : height;
  if (!Plot) return <div ref={box} style={{ height: h }} />;

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
      { x: fx, y: base.p10, mode: "lines", line: { width: 0, ...sl }, fill: "tonexty", fillpattern: { shape: "/", fgcolor: VG.grey, bgcolor: "rgba(255,255,255,0)", size: 9, solidity: 0.26 }, hoverinfo: "skip", name: "start-only band" },
      { x: fx, y: base.p50, mode: "lines", line: { color: VG.mute, width: 2, dash: "dash", ...sl }, hovertemplate: hov("start-only"), name: "start-only median" },
    );
  }
  // conditioned band
  data.push(
    { x: fx, y: cond.p90, mode: "lines", line: { width: 0, ...sl }, hoverinfo: "skip", showlegend: false },
    { x: fx, y: cond.p10, mode: "lines", line: { width: 0, ...sl }, fill: "tonexty", fillpattern: { shape: "/", fgcolor: VG.red, bgcolor: "rgba(255,255,255,0)", size: 9, solidity: 0.34 }, hoverinfo: "skip", name: "P10–P90" },
  );
  // sample paths
  cond.paths.slice(0, 6).forEach((p) => data.push({ x: fx, y: p, mode: "lines", line: { color: VG.red, width: 0.8, ...sl }, opacity: 0.35, hoverinfo: "skip", showlegend: false }));
  // conditioned mean (dotted) + median (solid)
  data.push(
    { x: fx, y: cond.mean, mode: "lines", line: { color: VG.black, width: 1.4, dash: "dot", ...sl }, hovertemplate: hov("mean"), name: "mean" },
    { x: fx, y: cond.p50, mode: "lines", line: { color: VG.red, width: 3.2, ...sl }, hovertemplate: hov("median"), name: "median" },
  );
  // conditioning context (history) up to day 0
  if (hx.length) data.push({ x: hx, y: history, mode: "lines", line: { color: VG.black, width: 3.2, ...sl }, hovertemplate: hov("history"), showlegend: false });

  const xmin = hx.length ? hxStart : 0;
  // y tick precision from the visible span, so small-range series (yields, spreads) never repeat a label
  const ys = [...cond.p10, ...cond.p90, ...(history ?? []), ...(showBase && base ? [...base.p10, ...base.p90] : [])].filter(Number.isFinite);
  const span = ys.length ? Math.max(...ys) - Math.min(...ys) : 1;
  const dec = span < 0.5 ? 2 : span < 5 ? 1 : 0;
  const xt = (narrow ? [xmin, 0, 15, 30] : [xmin, 0, 10, 20, 30]).filter((v, i, a) => v >= xmin && a.indexOf(v) === i);
  const xl = (v: number) => (v === 0 ? "Today" : v < 0 ? `${v}d` : `+${v}d`);
  const layout = {
    autosize: true, height: h, margin: { l: narrow ? 44 : 56, r: narrow ? 22 : 26, t: 28, b: 30 }, paper_bgcolor: VG.white, plot_bgcolor: VG.white,
    font: { family: VG_FONT, color: VG.mute, size: 12 },
    colorway: [VG.red, VG.black, VG.grey],
    hoverlabel: { bgcolor: VG.black, bordercolor: VG.black, font: { family: VG_FONT, color: VG.white, size: 12 } },
    xaxis: { range: [xmin, n - 1], tickvals: xt, ticktext: xt.map(xl), showgrid: false, zeroline: false, showline: true, linecolor: VG.black, linewidth: 2, ticks: "outside", tickcolor: VG.black, ticklen: 5, fixedrange: true },
    yaxis: { showgrid: true, gridcolor: VG.rule, zeroline: false, tickformat: `,.${dec}f`, nticks: narrow ? 5 : 7, fixedrange: true, tickprefix: "", ticksuffix: unit },
    showlegend: false,
    shapes: [
      ...(hx.length ? [{ type: "rect", x0: xmin, x1: 0, y0: 0, y1: 1, yref: "paper", fillcolor: VG.paper, line: { width: 0 }, layer: "below" }] : []),
      { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { color: VG.black, width: 2 } },
    ],
    annotations: [
      ...(hx.length ? [{ x: 0, y: 1, yref: "paper", xanchor: "right", yanchor: "bottom", xshift: -6, yshift: 4, text: narrow ? "<b>So far</b>" : "<b>Market so far</b>", showarrow: false, font: { color: VG.black, size: 12 } }] : []),
      { x: 0, y: 1, yref: "paper", xanchor: "left", yanchor: "bottom", xshift: 6, yshift: 4, text: narrow ? "<b>Next 30d</b>" : "<b>Next 30 days</b>", showarrow: false, font: { color: VG.red, size: 12 } },
    ],
    hovermode: "x unified",
  };
  void f;
  return (
    <div ref={box}>
      <Plot data={data as never} layout={layout as never} config={{ displayModeBar: false, responsive: true } as never} style={{ width: "100%", height: h }} useResizeHandler />
    </div>
  );
}
