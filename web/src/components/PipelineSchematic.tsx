import {
  createContext,
  Fragment,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactElement,
} from "react";
import {
  animate as animateValue,
  motion,
  useInView,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
  useTransform,
  type MotionValue,
} from "framer-motion";
import { cn } from "@/lib/utils";

/**
 * Animated summaries of the two ScenarioFlow workflows shown on the landing page.
 * The diagrams use the site tokens and resolve to static frames when the user
 * prefers reduced motion.
 */

// ── helpers (ported from the candidate gallery's cand-common) ────────────────
const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (e0: number, e1: number, x: number) => { const t = clamp((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const rnd = (n: number) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };
const randn = (seed: number, d: number) => {
  const u1 = clamp(rnd(seed * 1.7 + d * 0.311), 1e-4, 1), u2 = rnd(seed * 2.31 + d * 0.79);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.28318 * u2);
};
const fmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(2);
const POSITIVE = "oklch(0.62 0.15 152)";

const PipelineProgressContext = createContext<MotionValue<number> | null>(null);

// reduced-motion: system preference only
function useReduced() { return useReducedMotion() ?? false; }
// A block receives only its segment of the shared, one-shot pipeline progress.
function useProgress(reduced: boolean) {
  const contextProgress = useContext(PipelineProgressContext);
  const staticProgress = useMotionValue(1);
  const progress = contextProgress ?? staticProgress;
  return useTransform(progress, (v) => (reduced ? 1 : v));
}
// discrete integer step 0..steps from the shared sequence, frozen at `steps` when reduced
function useStep(steps: number, reduced: boolean, upTo = 0.7) {
  const [s, setS] = useState(reduced ? steps : 0);
  const progress = useProgress(reduced);
  const last = useRef(-1);
  useMotionValueEvent(progress, "change", (v) => {
    if (reduced) return;
    const k = Math.min(steps, Math.round(clamp(v / upTo) * steps));
    if (k !== last.current) { last.current = k; setS(k); }
  });
  return s;
}

// small quantile-fan builder (SCENARIO endpoint)
function buildFan(x0: number, x1: number, yt: number, yb: number, seed0 = 1, ns = 20) {
  const paths = Array.from({ length: ns }, (_, s) => {
    let v = 0; const a = [0];
    for (let d = 1; d <= 30; d++) { v += randn(s + seed0, d) - 0.025; a.push(v); }
    return a;
  });
  const allv = paths.flat(), vmin = Math.min(...allv) - 0.4, vmax = Math.max(...allv) + 0.4;
  const X = (d: number) => x0 + (d / 30) * (x1 - x0);
  const Y = (v: number) => yb - ((v - vmin) / (vmax - vmin)) * (yb - yt);
  const q = (arr: number[], p: number) => { const srt = [...arr].sort((a, b) => a - b); return srt[Math.floor(p * (srt.length - 1))]; };
  const Q = Array.from({ length: 31 }, (_, d) => { const c = paths.map((p) => p[d]); return { p10: q(c, 0.1), p50: q(c, 0.5), p90: q(c, 0.9) }; });
  const band = Q.map((qq, d) => `${X(d).toFixed(1)},${Y(qq.p90).toFixed(1)}`).concat(Q.map((qq, d) => `${X(d).toFixed(1)},${Y(qq.p10).toFixed(1)}`).reverse()).join(" ");
  const med = Q.map((qq, d) => `${X(d).toFixed(1)},${Y(qq.p50).toFixed(1)}`).join(" ");
  const line = (p: number[], k = 30) => p.slice(0, k + 1).map((v, d) => `${X(d).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  return { paths, band, med, line };
}

const VB = "0 0 264 150";
const svgStyle = { display: "block", overflow: "visible" } as const;

// ── ENCODE · text (encodeText·v4 "Letters → magnitude bars") ─────────────────
function ETBar({ p, i, n, reduced }: { p: MotionValue<number>; i: number; n: number; reduced: boolean }) {
  const h = 6 + Math.abs(randn(7, i + 1)) * 26;
  const sh = useTransform(p, (v) => (reduced ? h : lerp(2, h, smooth((i / n) * 0.5, (i / n) * 0.5 + 0.4, v))));
  const y = useTransform(sh, (v) => 96 - v);
  return <motion.rect x={20 + i * 9.5} width={6} rx={1.5} fill="var(--primary)" opacity={i % 3 === 0 ? 0.7 : 0.4} style={{ y, height: sh }} />;
}
function EncodeText() {
  const reduced = useReduced();
  const p = useProgress(reduced);
  const txtO = useTransform(p, (v) => (reduced ? 0 : clamp(1 - v * 2)));
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      <motion.text x={132} y={40} fontSize={13} fontWeight={600} textAnchor="middle" fill="var(--foreground)" style={{ fontFamily: "inherit", opacity: txtO }}>{`"dollar squeeze, risk-off"`}</motion.text>
      <line x1={18} y1={97} x2={246} y2={97} stroke="var(--border)" strokeWidth={1} />
      {Array.from({ length: 24 }).map((_, i) => <ETBar key={i} p={p} i={i} n={24} reduced={reduced} />)}
      <text x={18} y={116} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>text representation</text>
    </svg>
  );
}

// ── ENCODE · numeric (encodeNumeric·v2; factor rows → structured sidecar) ────
const EN_FAC = [
  { k: "SPX", v: -1.8, seed: 3 },
  { k: "UST10Y", v: 0.9, seed: 8 },
  { k: "USD", v: 1.4, seed: 5 },
  { k: "HY OAS", v: 2.1, seed: 11 },
];
const enSpark = (seed: number, x0: number, y0: number, w: number, h: number, n = 18) => {
  let v = 0; const ys = [0];
  for (let i = 1; i < n; i++) { v += randn(seed, i); ys.push(v); }
  const mn = Math.min(...ys), mx = Math.max(...ys), sp = mx - mn || 1;
  return ys.map((y, i) => `${(x0 + (i / (n - 1)) * w).toFixed(1)},${(y0 + h - ((y - mn) / sp) * h).toFixed(1)}`).join(" ");
};
function EncodeNumeric() {
  const reduced = useReduced();
  const step = useStep(EN_FAC.length, reduced, 0.8);
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      {EN_FAC.map((f, i) => {
        const on = reduced || i <= step;
        return (
          <g key={i} transform={`translate(14 ${30 + i * 26})`}>
            <text x={0} y={4} fontSize={8.5} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>{f.k}</text>
            <polyline points={enSpark(f.seed, 48, -7, 50, 16)} fill="none" stroke="var(--primary)" strokeWidth={1.2} strokeOpacity={on ? 0.7 : 0.25} />
            <rect x={108} y={-7} width={34} height={15} rx={3} fill={on ? "color-mix(in oklab, var(--primary) 14%, var(--card))" : "var(--secondary)"} stroke="var(--border)" strokeWidth={1} />
            <text x={125} y={3.5} fontSize={8.5} fontWeight={700} textAnchor="middle" fill={on ? "var(--foreground)" : "var(--muted-foreground)"} style={{ fontFamily: "ui-monospace, monospace", opacity: on ? 1 : 0.4 }}>{fmt(f.v)}σ</text>
            <line x1={146} y1={0} x2={168} y2={0} stroke="var(--border)" strokeWidth={1} opacity={on ? 0.8 : 0.3} />
            <rect x={170} y={-6} width={12} height={12} rx={2} fill="var(--primary)" opacity={on ? (i % 2 ? 0.4 : 0.72) : 0.1} />
          </g>
        );
      })}
      <text x={166} y={132} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>market summary</text>
    </svg>
  );
}

// ── PROJECT (project·v2 "Coordinate remap" — learned regression map) ─────────
function Project() {
  const reduced = useReduced();
  const progress = useProgress(reduced);
  const t = useTransform(progress, (v) => easeInOut(clamp((v - 0.05) / 0.8)));
  const bez = (tt: number): [number, number] => {
    const u = 1 - tt;
    const x = u * u * u * 54 + 3 * u * u * tt * 96 + 3 * u * tt * tt * 150 + tt * tt * tt * 196;
    const y = u * u * u * 64 + 3 * u * u * tt * 30 + 3 * u * tt * tt * 96 + tt * tt * tt * 74;
    return [x, y];
  };
  const cx = useTransform(t, (v) => bez(v)[0]);
  const cy = useTransform(t, (v) => bez(v)[1]);
  const warp = (gx: number, gy: number): [number, number] => [gx, gy + Math.sin((gx - 150) / 22) * 6 * ((gy - 40) / 60)];
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      {[0, 1, 2, 3].map((i) => <line key={`lh${i}`} x1={20} y1={40 + i * 20} x2={100} y2={40 + i * 20} stroke="var(--border)" strokeWidth={0.8} />)}
      {[0, 1, 2, 3, 4].map((i) => <line key={`lv${i}`} x1={20 + i * 20} y1={40} x2={20 + i * 20} y2={100} stroke="var(--border)" strokeWidth={0.8} />)}
      <text x={20} y={32} fontSize={7.5} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>language</text>
      <circle cx={54} cy={64} r={3} fill="var(--muted-foreground)" opacity={0.6} />
      {[0, 1, 2, 3].map((r) => <polyline key={`rh${r}`} points={[150, 170, 190, 210, 230].map((gx) => warp(gx, 40 + r * 20).map((n) => n.toFixed(1)).join(",")).join(" ")} fill="none" stroke="var(--border)" strokeWidth={0.8} />)}
      {[150, 170, 190, 210, 230].map((gx, i) => <polyline key={`rv${i}`} points={[40, 60, 80, 100].map((gy) => warp(gx, gy).map((n) => n.toFixed(1)).join(",")).join(" ")} fill="none" stroke="var(--border)" strokeWidth={0.8} />)}
      <text x={150} y={32} fontSize={7.5} fontWeight={600} fill="var(--primary)" style={{ fontFamily: "inherit" }}>market history</text>
      <path d="M 60 64 C 96 30, 150 96, 196 74" fill="none" stroke="var(--primary)" strokeWidth={1.2} strokeDasharray="3 3" opacity={0.45} />
      <motion.circle r={3.4} fill="var(--primary)" style={{ cx, cy }} />
      <text x={20} y={134} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>connect language to market history</text>
    </svg>
  );
}

// ── EMBEDDING MANIFOLD (manifold·v1 "Radial retrieve") ───────────────────────
const MF_QX = 120, MF_QY = 70;
const MF_BANK = Array.from({ length: 42 }, (_, i) => ({ x: 16 + rnd(i + 1) * 198, y: 22 + rnd(i + 7.3) * 96 }));
const MF_DIST = MF_BANK.map((p, i) => ({ i, d: Math.hypot(p.x - MF_QX, p.y - MF_QY) })).sort((a, b) => a.d - b.d);
const MF_NEAR = [MF_DIST[1], MF_DIST[2], MF_DIST[3]].map((o, k) => ({ ...MF_BANK[o.i], w: [0.6, 0.3, 0.1][k] }));
function Manifold() {
  const reduced = useReduced();
  const progress = useProgress(reduced);
  const R = useTransform(progress, (v) => (reduced ? 78 : lerp(0, 78, easeInOut(clamp((v - 0.05) / 0.7)))));
  const ringO = useTransform(R, (v) => clamp(1 - v / 110) * 0.6);
  // MF_NEAR is a module constant (fixed length 3) → hook order is stable across renders.
  const op0 = useTransform(R, (v) => (v >= Math.hypot(MF_NEAR[0].x - MF_QX, MF_NEAR[0].y - MF_QY) - 1 ? 1 : 0));
  const op1 = useTransform(R, (v) => (v >= Math.hypot(MF_NEAR[1].x - MF_QX, MF_NEAR[1].y - MF_QY) - 1 ? 1 : 0));
  const op2 = useTransform(R, (v) => (v >= Math.hypot(MF_NEAR[2].x - MF_QX, MF_NEAR[2].y - MF_QY) - 1 ? 1 : 0));
  const ops = [op0, op1, op2];
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      {MF_BANK.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={1.6} fill="var(--muted-foreground)" opacity={0.3} />)}
      <motion.circle cx={MF_QX} cy={MF_QY} fill="none" stroke="var(--primary)" strokeWidth={1.1} style={{ r: R, opacity: ringO }} />
      {MF_NEAR.map((n, k) => (
        <motion.g key={k} style={{ opacity: ops[k] }}>
          <circle cx={n.x} cy={n.y} r={4 + n.w * 12} fill="var(--primary)" opacity={0.14} />
          <circle cx={n.x} cy={n.y} r={2.6} fill="var(--primary)" />
          <text x={n.x} y={n.y - 7 - n.w * 12} fontSize={7.5} fontWeight={700} textAnchor="middle" fill="var(--primary)" style={{ fontFamily: "inherit" }}>{n.w.toFixed(1)}</text>
        </motion.g>
      ))}
      <circle cx={MF_QX} cy={MF_QY} r={3.4} fill="var(--foreground)" />
      <text x={MF_QX} y={MF_QY + 16} fontSize={7.5} fontWeight={600} textAnchor="middle" fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>description</text>
      <text x={16} y={14} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>related historical periods</text>
    </svg>
  );
}

// ── LANGUAGE MODEL ──────────────────────────────────────────────────────────
function AUCaret({ x, y, reduced }: { x: number; y: number; reduced: boolean }) {
  return <motion.rect x={x} y={y - 7} width={2} height={11} rx={1} fill="var(--primary)"
    animate={{ opacity: 1 }}
    transition={reduced ? { duration: 0 } : { duration: 0.18 }} />;
}
function Author() {
  const reduced = useReduced();
  const step = useStep(4, reduced, 0.85);
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      <rect x={14} y={22} width={236} height={104} rx={8} fill="color-mix(in oklab, var(--foreground) 4%, var(--card))" stroke="var(--border)" strokeWidth={1} />
      <circle cx={26} cy={34} r={2.4} fill="var(--destructive)" opacity={0.5} />
      <circle cx={34} cy={34} r={2.4} fill="var(--muted-foreground)" opacity={0.4} />
      <circle cx={42} cy={34} r={2.4} fill="var(--muted-foreground)" opacity={0.4} />
      <text x={232} y={37} fontSize={7.5} fontWeight={600} textAnchor="end" fill="var(--muted-foreground)" style={{ fontFamily: "ui-monospace, monospace" }}>language model</text>
      <text x={26} y={56} fontSize={8} fontWeight={600} fill="var(--primary)" style={{ fontFamily: "inherit" }}>summarize market movements</text>
      {[120, 150, 96].map((w, i) => {
        const on = reduced || i < step, active = !reduced && i === step;
        return (
          <g key={i}>
            <rect x={26} y={66 + i * 14} width={w} height={4.5} rx={2.2} fill="var(--foreground)" opacity={on ? 0.32 : active ? 0.14 : 0.06} />
            {active && <AUCaret x={26 + w + 3} y={66 + i * 14 + 2} reduced={reduced} />}
          </g>
        );
      })}
      <text x={26} y={117} fontSize={7.5} fontWeight={600} fill="var(--destructive)" style={{ fontFamily: "ui-monospace, monospace", opacity: 0.8 }}>contrasting interpretation</text>
      <text x={14} y={140} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>The language model summarizes the episode</text>
    </svg>
  );
}

// ── endpoints (no candidate picked — kept simple) ────────────────────────────
function NarrativeCard({ positive = false }: { positive?: boolean }) {
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      <rect x={18} y={30} width={228} height={92} rx={10} fill="color-mix(in oklab, var(--primary) 5%, var(--card))" stroke="var(--border)" strokeWidth={1} />
      <text x={34} y={72} fontSize={15} fontWeight={600} fill="var(--foreground)" style={{ fontFamily: "inherit" }}>{`"dollar squeeze,`}</text>
      <text x={34} y={94} fontSize={15} fontWeight={600} fill="var(--foreground)" style={{ fontFamily: "inherit" }}>{`risk-off"`}</text>
      <g transform="translate(18 18)">
        <rect x={0} y={-9} width={56} height={16} rx={8}
          fill={positive ? `color-mix(in oklab, ${POSITIVE} 16%, var(--card))` : "var(--secondary)"}
          stroke={positive ? POSITIVE : "var(--border)"} strokeWidth={1} strokeOpacity={positive ? 0.6 : 1} />
        <text x={28} y={3} fontSize={8} fontWeight={700} textAnchor="middle"
          fill={positive ? POSITIVE : "var(--muted-foreground)"} style={{ fontFamily: "inherit" }}>{positive ? "main" : "input"}</text>
      </g>
    </svg>
  );
}
function ScenarioFan() {
  const reduced = useReduced();
  const progress = useProgress(reduced);
  const F = buildFan(20, 244, 30, 118, 7, 18);
  const breathe = useTransform(progress, (v) => (reduced ? 0.18 : lerp(0.04, 0.18, smooth(0, 0.7, v))));
  return (
    <svg viewBox={VB} width="100%" style={svgStyle}>
      <motion.polygon points={F.band} fill="var(--primary)" style={{ opacity: breathe }} />
      {F.paths.slice(0, 7).map((pa, i) => <polyline key={i} points={F.line(pa)} fill="none" stroke="var(--primary)" strokeWidth={0.7} strokeOpacity={0.32} />)}
      <polyline points={F.med} fill="none" stroke="var(--primary)" strokeWidth={1.6} />
      <text x={20} y={140} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>30-day conditional distribution</text>
    </svg>
  );
}

const OBSERVED_PATHS = [
  { label: "SPX", value: "−1.8σ", opacity: 0.95, points: "58,30 72,28 86,32 100,31 114,36 128,34 142,38 156,37 170,42 184,40 198,45 212,43 228,48" },
  { label: "VIX", value: "+1.4σ", opacity: 0.78, points: "58,62 72,64 86,59 100,61 114,55 128,57 142,51 156,53 170,47 184,49 198,42 212,44 228,38" },
  { label: "UST10Y", value: "+0.9σ", opacity: 0.64, points: "58,84 72,82 86,85 100,80 114,81 128,78 142,79 156,75 170,77 184,73 198,74 212,71 228,72" },
  { label: "USD", value: "+1.1σ", opacity: 0.5, points: "58,111 72,108 86,110 100,106 114,107 128,102 142,104 156,99 170,101 184,96 198,97 212,92 228,94" },
] as const;

function ObservedScenario() {
  return (
    <svg viewBox={VB} width="100%" style={svgStyle} role="img" aria-label="Observed 30-day market episode">
      <title>Observed 30-day market episode</title>
      <rect x={12} y={16} width={240} height={110} rx={8} fill="color-mix(in oklab, var(--foreground) 3%, var(--card))" stroke="var(--border)" strokeWidth={1} />
      {OBSERVED_PATHS.map((path, index) => {
        const baseline = 34 + index * 26;
        return (
          <g key={path.label}>
            <text x={20} y={baseline + 3} fontSize={7.5} fontWeight={700} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>{path.label}</text>
            <line x1={56} y1={baseline} x2={230} y2={baseline} stroke="var(--border)" strokeWidth={0.7} />
            <polyline points={path.points} fill="none" stroke="var(--primary)" strokeWidth={1.5} strokeOpacity={path.opacity} strokeLinecap="round" strokeLinejoin="round" />
            <text x={244} y={baseline + 3} fontSize={7.5} fontWeight={700} textAnchor="end" fill="var(--foreground)" style={{ fontFamily: "ui-monospace, monospace" }}>{path.value}</text>
          </g>
        );
      })}
      <text x={14} y={140} fontSize={8} fontWeight={600} fill="var(--muted-foreground)" style={{ fontFamily: "inherit" }}>observed 30-day market paths</text>
    </svg>
  );
}

// ── flow definitions ─────────────────────────────────────────────────────────
type Block = { label: string; C: () => ReactElement };
type Flow = { caption: string; blocks: Block[] };

const NarrativeInput = () => <NarrativeCard />;
const NarrativeOutput = () => <NarrativeCard positive />;

const FLOWS: Record<"narrative-to-scenario" | "scenario-to-narrative", Flow> = {
  "narrative-to-scenario": {
    caption: "ScenarioFlow links the description to related historical conditions and generates a conditional distribution.",
    blocks: [
      { label: "NARRATIVE", C: NarrativeInput },
      { label: "TEXT", C: EncodeText },
      { label: "CONNECT TO HISTORY", C: Project },
      { label: "RELATED HISTORY", C: Manifold },
      { label: "SCENARIOFLOW OUTPUT", C: ScenarioFan },
    ],
  },
  "scenario-to-narrative": {
    caption: "The demo turns observed market movements into a plain-language narrative and a control interpretation.",
    blocks: [
      { label: "OBSERVED EPISODE", C: ObservedScenario },
      { label: "MARKET MOVEMENTS", C: EncodeNumeric },
      { label: "LANGUAGE MODEL", C: Author },
      { label: "NARRATIVE", C: NarrativeOutput },
    ],
  },
};

function Arrow({ revealed }: { revealed: boolean }) {
  return (
    <motion.div
      animate={{ opacity: revealed ? 1 : 0.25 }}
      className="flex h-7 shrink-0 items-center justify-center text-muted-foreground/55 md:h-auto md:w-5 md:pt-[30px]"
      transition={{ duration: 0.18 }}
    >
      <svg className="rotate-90 md:rotate-0" width="26" height="16" viewBox="0 0 26 16" aria-hidden>
        <line x1={1} y1={8} x2={20} y2={8} stroke="currentColor" strokeWidth={1.1} />
        <path d="M17 4.5 L21.5 8 L17 11.5" fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </motion.div>
  );
}

function PipelineBlock({
  block,
  index,
  count,
  progress,
  activeStep,
  isRunning,
  reduced,
}: {
  block: Block;
  index: number;
  count: number;
  progress: MotionValue<number>;
  activeStep: number;
  isRunning: boolean;
  reduced: boolean;
}) {
  const localProgress = useTransform(progress, [index / count, (index + 1) / count], [0, 1]);
  const revealed = reduced || activeStep >= index;
  const active = isRunning && activeStep === index;

  return (
    <motion.div
      animate={{ opacity: revealed ? 1 : 0.38, y: revealed ? 0 : 4 }}
      className="flex w-full flex-col md:min-w-0 md:flex-1"
      transition={reduced ? { duration: 0 } : { duration: 0.18 }}
    >
      <div className="mb-1 flex min-h-[26px] items-end justify-center px-0.5 text-center text-[10px] font-semibold uppercase leading-tight tracking-wider text-muted-foreground">
        {block.label}
      </div>
      <motion.div
        animate={{
          borderColor: active ? "var(--primary)" : "var(--border)",
          boxShadow: active ? "0 0 0 2px color-mix(in oklab, var(--primary) 12%, transparent)" : "0 0 0 0 transparent",
        }}
        className="mx-auto flex min-h-0 w-full max-w-[320px] flex-1 items-center rounded-xl border bg-card/30 p-1.5 md:max-w-none"
        transition={reduced ? { duration: 0 } : { duration: 0.18 }}
      >
        <PipelineProgressContext.Provider value={localProgress}>
          <block.C />
        </PipelineProgressContext.Provider>
      </motion.div>
    </motion.div>
  );
}

export default function PipelineSchematic({ direction, className }: { direction: "narrative-to-scenario" | "scenario-to-narrative"; className?: string }) {
  const flow = FLOWS[direction];
  const reduced = useReduced();
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef, { once: true, amount: 0.35 });
  const progress = useMotionValue(reduced ? 1 : 0);
  const controlsRef = useRef<ReturnType<typeof animateValue> | null>(null);
  const playedRef = useRef(reduced);
  const [activeStep, setActiveStep] = useState(reduced ? flow.blocks.length : -1);
  const [isRunning, setIsRunning] = useState(false);

  useMotionValueEvent(progress, "change", (value) => {
    if (reduced) return;
    setActiveStep(Math.min(flow.blocks.length - 1, Math.floor(value * flow.blocks.length)));
  });

  const play = useCallback(() => {
    controlsRef.current?.stop();
    if (reduced) {
      progress.set(1);
      setActiveStep(flow.blocks.length);
      return;
    }
    progress.set(0);
    setActiveStep(0);
    setIsRunning(true);
    controlsRef.current = animateValue(progress, 1, {
      duration: flow.blocks.length * 0.58,
      ease: "easeInOut",
      onComplete: () => {
        setActiveStep(flow.blocks.length);
        setIsRunning(false);
      },
    });
  }, [flow.blocks.length, progress, reduced]);

  useEffect(() => {
    if (!inView || playedRef.current) return;
    playedRef.current = true;
    play();
  }, [inView, play]);

  useEffect(() => () => {
    controlsRef.current?.stop();
    controlsRef.current = null;
    playedRef.current = false;
  }, []);

  return (
    <div ref={rootRef} className={cn("w-full", className)}>
      <div className="flex flex-col items-stretch md:flex-row md:gap-1">
          {flow.blocks.map((b, i) => (
            <Fragment key={b.label + i}>
              <PipelineBlock
                block={b}
                index={i}
                count={flow.blocks.length}
                progress={progress}
                activeStep={activeStep}
                isRunning={isRunning}
                reduced={reduced}
              />
              {i < flow.blocks.length - 1 && <Arrow revealed={reduced || activeStep > i} />}
            </Fragment>
          ))}
      </div>
      <div className="mt-3 flex flex-col items-center justify-between gap-2 sm:flex-row">
        <p className="text-center text-xs leading-relaxed text-muted-foreground sm:text-left">{flow.caption}</p>
        <button
          type="button"
          onClick={play}
          disabled={reduced || isRunning}
          className="inline-flex min-h-9 shrink-0 items-center rounded-full border border-border px-3 text-xs font-medium text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-default disabled:opacity-45"
          aria-label={`Replay ${direction === "narrative-to-scenario" ? "narrative to scenario" : "scenario to narrative"} sequence`}
        >
          ↻&nbsp; Replay
        </button>
      </div>
    </div>
  );
}
