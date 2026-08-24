// Tiny inline SVG sparkline for a factor's 30-day movement (no charting dep).
export function Sparkline({ values, width = 96, height = 24 }: { values?: number[]; width?: number; height?: number }) {
  if (!values || values.length < 2) return <span className="text-xs text-muted-foreground">—</span>;
  const min = Math.min(...values), max = Math.max(...values), rng = max - min || 1;
  const x = (i: number) => (i / (values.length - 1)) * (width - 2) + 1;
  const y = (v: number) => height - 2 - ((v - min) / rng) * (height - 4);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const up = values[values.length - 1] >= values[0];
  const color = up ? "#1a7f37" : "#d70015";
  return (
    <svg width={width} height={height} className="block">
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.3} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r={1.6} fill={color} />
    </svg>
  );
}
