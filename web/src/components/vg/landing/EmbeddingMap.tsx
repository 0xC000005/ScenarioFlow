import { useEffect, useState } from "react";
import { VG } from "@/lib/palette";
import { fmtDate } from "./shift";

// Where the 20 Jan 2016 words land in the model's memory. This is the SAME 2-D map as the story chapter
// "What the model remembers" (t-SNE of the 4,010 past 30-day windows together with the bridge-projected query;
// data_v4.js V4.mem), on the same
// 0–1 coordinates with the story's 1 : 0.8 aspect. The three windows the run drew on carry the run's own matches.

type Drawn = { i: number; date: string; cos: number; xy: [number, number] };
type MemMap = {
  story: string;
  n: number;
  first: string;
  last: string;
  q: [number, number];
  pts: [number, number, 0 | 1][];
  drawn: Drawn[];
};

const SRC = "/data/memory_map_story.json";
const W = 640, H = 512, PAD = 16;
const sx = (x: number) => PAD + x * (W - 2 * PAD);
const sy = (y: number) => PAD + y * (H - 2 * PAD);

export default function EmbeddingMap() {
  const [d, setD] = useState<MemMap | null>(null);
  const [failed, setFailed] = useState(false);
  const [hot, setHot] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(SRC)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json() as Promise<MemMap>;
      })
      .then((x) => alive && setD(x))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, []);

  if (failed) return <p className="text-[15px] text-vg-mute">The memory map could not load.</p>;
  if (!d) return <div className="min-h-[420px]" />;
  const [qx, qy] = d.q;

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-12">
      <figure className="min-w-0">
        <div className="border-l-[10px] border-vg-red bg-vg-black">
          <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img"
            aria-label={`${d.n.toLocaleString("en-US")} past markets from ${d.first.slice(0, 4)} to ${d.last.slice(0, 4)} as the model remembers them; stressed markets in red; the 20 January 2016 words as a red diamond, linked to the ${d.drawn.length} past markets the model drew on`}>
            {d.pts.map(([x, y, s], i) => (
              <circle key={i} cx={sx(x)} cy={sy(y)} r={s ? 2.4 : 1.9} fill={s ? VG.red : "#6E6E6E"} opacity={s ? 0.85 : 0.55} />
            ))}
            {d.drawn.map((n, k) => (
              <line key={"l" + n.i} x1={sx(qx)} y1={sy(qy)} x2={sx(n.xy[0])} y2={sy(n.xy[1])}
                stroke={VG.white} strokeWidth={hot === k ? 3.5 : 2} />
            ))}
            {d.drawn.map((n, k) => (
              <g key={"n" + n.i}>
                <circle cx={sx(n.xy[0])} cy={sy(n.xy[1])} r={hot === k ? 11 : 9.5} fill={VG.black} stroke={VG.white} strokeWidth={2} />
                <text x={sx(n.xy[0])} y={sy(n.xy[1]) + 4.5} textAnchor="middle" fontSize={12.5} fontWeight={700} fill={VG.white} fontFamily="Inter">
                  {k + 1}
                </text>
              </g>
            ))}
            <rect x={sx(qx) - 8} y={sy(qy) - 8} width={16} height={16} fill={VG.red} stroke={VG.white} strokeWidth={2}
              transform={`rotate(45 ${sx(qx)} ${sy(qy)})`} />
          </svg>
        </div>
        <figcaption className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-[14px] font-semibold">
          <span className="inline-flex items-center gap-2 text-vg-red">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><rect x="3" y="3" width="10" height="10" fill={VG.red} transform="rotate(45 8 8)" /></svg>
            your words
          </span>
          <span className="inline-flex items-center gap-2 text-vg-mute">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="3" fill="#6E6E6E" /></svg>
            past markets, {d.first.slice(0, 4)}–{d.last.slice(0, 4)}
          </span>
          <span className="inline-flex items-center gap-2 text-vg-mute">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="3.5" fill={VG.red} /></svg>
            stressed (VIX above 28)
          </span>
        </figcaption>
      </figure>

      <div className="min-w-0">
        <p className="text-[15px] font-semibold text-vg-black">The past markets the model drew on</p>
        <ol className="mt-3 border-b-2 border-vg-black">
          {d.drawn.map((n, k) => (
            <li key={n.i} className="border-t-2 border-vg-black" onMouseEnter={() => setHot(k)} onMouseLeave={() => setHot(null)}>
              <div className="flex items-baseline gap-x-3 py-2.5">
                <span className="grid h-6 w-6 shrink-0 place-items-center self-center bg-vg-black text-[12px] font-bold text-vg-white">{k + 1}</span>
                <span className="whitespace-nowrap text-[17px] font-bold">{fmtDate(n.date)}</span>
                <span className="ml-auto whitespace-nowrap text-[14px] tabular-nums text-vg-mute">match {n.cos.toFixed(2)}</span>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
