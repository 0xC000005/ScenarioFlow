import { motion } from "framer-motion";
import { VG } from "@/lib/palette";

type Tone = "black" | "red" | "grey" | "white";
const STROKE: Record<Tone, string> = { black: VG.black, red: VG.red, grey: VG.grey, white: VG.white };

// Inline SVG sparkline for a factor's observed 30-day movement (real path, no charting dep).
// `drawDelay` (seconds) draws the line on mount; pass undefined for a static line.
export function Sparkline({
  values,
  width = 96,
  height = 24,
  tone = "black",
  drawDelay,
}: {
  values?: number[];
  width?: number;
  height?: number;
  tone?: Tone;
  drawDelay?: number;
}) {
  if (!values || values.length < 2) return <span className="text-xs text-vg-grey">—</span>;
  const min = Math.min(...values), max = Math.max(...values), rng = max - min || 1;
  const pad = 3;
  const x = (i: number) => (i / (values.length - 1)) * (width - 2 * pad) + pad;
  const y = (v: number) => height - pad - ((v - min) / rng) * (height - 2 * pad);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const color = STROKE[tone];
  const animated = drawDelay !== undefined;
  const ex = x(values.length - 1), ey = y(values[values.length - 1]);
  return (
    <svg width={width} height={height} className="block" aria-hidden="true">
      <line x1={pad} x2={pad} y1={pad} y2={height - pad} stroke={VG.soft} strokeWidth={1} />
      <motion.polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={tone === "red" ? 2.2 : 1.6}
        strokeLinejoin="miter"
        strokeLinecap="square"
        initial={animated ? { pathLength: 0 } : false}
        animate={{ pathLength: 1 }}
        transition={animated ? { duration: 0.7, delay: drawDelay, ease: "easeOut" } : { duration: 0 }}
      />
      <motion.rect
        x={ex - 2.5}
        y={ey - 2.5}
        width={5}
        height={5}
        fill={color}
        initial={animated ? { opacity: 0 } : false}
        animate={{ opacity: 1 }}
        transition={animated ? { duration: 0.15, delay: (drawDelay ?? 0) + 0.7 } : { duration: 0 }}
      />
    </svg>
  );
}
