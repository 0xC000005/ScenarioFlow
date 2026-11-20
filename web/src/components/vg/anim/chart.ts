// Small, dependency-free helpers for the hand-drawn SVG charts in the landing-page animations.

export type Scale = (v: number) => number;

export const linear = (d0: number, d1: number, r0: number, r1: number): Scale => (v) =>
  r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

/** ~n "nice" ticks covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Polyline path through (x(i), y(v)) for the first `upto` points (fractional = partial last segment). */
export function linePath(vals: number[], x: (i: number) => number, y: Scale, upto = vals.length - 1): string {
  if (!vals.length) return "";
  const end = Math.max(0, Math.min(vals.length - 1, upto));
  const whole = Math.floor(end);
  let d = `M${x(0).toFixed(1)},${y(vals[0]).toFixed(1)}`;
  for (let i = 1; i <= whole; i++) d += `L${x(i).toFixed(1)},${y(vals[i]).toFixed(1)}`;
  const f = end - whole;
  if (f > 0 && whole + 1 < vals.length) {
    const v = vals[whole] + (vals[whole + 1] - vals[whole]) * f;
    d += `L${x(whole + f).toFixed(1)},${y(v).toFixed(1)}`;
  }
  return d;
}

/** Closed band between `lo` and `hi` for the first `upto` points. */
export function bandPath(lo: number[], hi: number[], x: (i: number) => number, y: Scale, upto = lo.length - 1): string {
  const top = linePath(hi, x, y, upto);
  const end = Math.max(0, Math.min(lo.length - 1, upto));
  const whole = Math.floor(end);
  const f = end - whole;
  const pts: string[] = [];
  if (f > 0 && whole + 1 < lo.length) {
    const v = lo[whole] + (lo[whole + 1] - lo[whole]) * f;
    pts.push(`${x(whole + f).toFixed(1)},${y(v).toFixed(1)}`);
  }
  for (let i = whole; i >= 0; i--) pts.push(`${x(i).toFixed(1)},${y(lo[i]).toFixed(1)}`);
  return `${top}L${pts.join("L")}Z`;
}

export const mix = (a: number[], b: number[], k: number) => a.map((v, i) => v + (b[i] - v) * k);

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "20 Jan 2016" (or "20 Jan" without the year), from an ISO date, locale-independent. */
export function fmtDate(iso: string, withYear = true) {
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]}${withYear ? ` ${y}` : ""}`;
}
