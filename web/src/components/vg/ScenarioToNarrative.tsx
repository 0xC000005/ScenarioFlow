import { useRef } from "react";
import { VG } from "@/lib/palette";
import { useJsonResource, useTimeline, useWidth, seg, easeOut, easeInOut } from "@/components/vg/anim/timeline";
import { Typed, ReplayButton, FigureLoadError } from "@/components/vg/anim/Controls";
import { fmtDate, linear, linePath } from "@/components/vg/anim/chart";

// Real data only: public/data/vg/scenario_to_narrative_2008-10-31.json, extracted unchanged from the cached
// scenario-to-narrative demo output for the window ending 31 Oct 2008 (see its "source" field):
// factor paths = observed closes 22 Sep – 31 Oct 2008; z = move size in standard deviations from the sidecar's
// mechanical summary; the narrative shown is the workbench's risk-manager memo (positive narrative, view
// "risk_manager_memo"). The file also holds other views and hard negatives; the landing shows only the memo.

type Factor = { key: string; label: string; unit: string; start: number; end: number; delta: number; z: number; path: number[] };
type View = { view: string; label: string; text: string };
type Episode = { episode: { start_date: string; end_date: string; window_id: string }; factors: Factor[]; views: View[] };

const TOP = 4; // largest moves (|z|) shown heavier, in black: red is reserved for words / the language-model memo
const LIST = 6; // moves kept in the short list

const sign = (v: number) => (v < 0 ? "−" : "+");
// Direction is carried by the glyph (and the sign); move size by line/bar weight, never by colour.
const glyph = (v: number) => (v > 0 ? "▲" : v < 0 ? "▼" : "■");
function fmtDelta(f: Factor) {
  const a = Math.abs(f.delta);
  switch (f.unit) {
    case "bp": return `${sign(f.delta)}${(a * 100).toFixed(0)} bp`;
    case "usd": return `${sign(f.delta)}$${a < 100 ? a.toFixed(1) : a.toFixed(0)}`;
    case "yen": return `${sign(f.delta)}${a.toFixed(1)} yen`;
    default: return `${sign(f.delta)}${a < 100 ? a.toFixed(1) : a.toFixed(0)}${f.key === "DXY" ? "" : " pts"}`;
  }
}
function fmtLevel(f: Factor, v: number) {
  if (f.unit === "bp") return `${v.toFixed(2)}%`;
  if (f.unit === "usd") return `$${v.toFixed(v < 200 ? 1 : 0)}`;
  return v >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : v.toFixed(v < 10 ? 2 : 1);
}
const range = (a: string, b: string) =>
  `${fmtDate(a, false)} – ${fmtDate(b)}`;

const DURATION = 13;
const P = { draw: [0, 0.3], mark: [0.3, 0.36], list: [0.34, 0.5], text: [0.52, 0.98] } as const;
const MEMO_VIEW = "risk_manager_memo";

function Spark({ f, p, hot }: { f: Factor; p: number; hot: boolean }) {
  const [box, W] = useWidth<HTMLDivElement>(180);
  const H = 80;
  const lo = Math.min(...f.path), hi = Math.max(...f.path);
  const y = linear(lo, hi, H - 4, 4);
  const x = (i: number) => 2 + (i / (f.path.length - 1)) * (W - 8);
  const upto = easeInOut(p) * (f.path.length - 1);
  return (
    <div ref={box} className="w-full">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} aria-hidden>
        <line x1={2} x2={W - 6} y1={y(f.start)} y2={y(f.start)} stroke={VG.rule} strokeWidth={1} />
        <path d={linePath(f.path, x, y, upto)} fill="none" stroke={hot ? VG.black : VG.mute} strokeWidth={hot ? 3 : 1.6} strokeLinejoin="round" />
      </svg>
    </div>
  );
}

export default function ScenarioToNarrative() {
  const { data, failed, retry } = useJsonResource<Episode>("/data/vg/scenario_to_narrative_2008-10-31.json");
  const root = useRef<HTMLElement>(null);
  // poster = sparklines drawn, largest movers marked and ranked; the narrative then types over its ghost
  const tl = useTimeline(root, DURATION, !!data, P.text[0]);
  const label = data
    ? `Scenario to narrative: observed market moves ${range(data.episode.start_date, data.episode.end_date)} and the narrative written from them`
    : "Scenario to narrative";
  return (
    <figure ref={root} className="w-full bg-vg-white p-5 text-vg-black sm:p-8" aria-label={label} aria-busy={!data && !failed}>
      {data ? <Body data={data} t={tl.t} replay={tl.replay} /> : failed ? <FigureLoadError onRetry={retry} minHeight={320} /> : <div className="min-h-[640px]" />}
    </figure>
  );
}

function Body({ data, t, replay }: { data: Episode; t: number; replay: () => void }) {

  const ranked = [...data.factors].sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
  const hotKeys = new Set(ranked.slice(0, TOP).map((f) => f.key));
  const marked = t >= P.mark[0];
  const maxZ = Math.abs(ranked[0].z);
  const view = data.views.find((v) => v.view === MEMO_VIEW) ?? data.views[0];
  const ep = data.episode;

  return (
    <>
      <div className="grid gap-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,4fr)] lg:gap-12">
        {/* observed episode as small multiples */}
        <div className="min-w-0">
          <p className="text-[15px] font-semibold">Observed market, {range(ep.start_date, ep.end_date)}</p>
          <ul className="mt-3 grid grid-cols-2 gap-x-5 gap-y-5 sm:grid-cols-3 md:grid-cols-5">
            {data.factors.map((f, i) => {
              const p = seg(t, P.draw[0] + i * 0.012, P.draw[1] - (data.factors.length - 1 - i) * 0.006);
              const hot = marked && hotKeys.has(f.key);
              return (
                <li key={f.key} className={"border-t-[3px] pt-2 transition-colors duration-500 " + (hot ? "border-vg-black" : "border-vg-rule")}>
                  <p className="text-[14px] font-semibold leading-tight">{f.label}</p>
                  <p className={"text-[13px] tabular-nums transition-colors duration-500 " + (hot ? "font-bold text-vg-black" : "text-vg-mute")} style={{ opacity: p }}>
                    <span aria-hidden className="mr-1 text-[10px]">{glyph(f.delta)}</span>{fmtDelta(f)}
                  </p>
                  <Spark f={f} p={p} hot={hot} />
                  <p className="text-[11px] tabular-nums text-vg-grey">
                    {fmtLevel(f, f.start)} to {fmtLevel(f, f.end)}
                  </p>
                </li>
              );
            })}
          </ul>
        </div>

        {/* compressed: the largest moves */}
        <div className="min-w-0">
          <p className="text-[15px] font-semibold">Largest moves, in standard deviations</p>
          <ol className="mt-3 border-t-[3px] border-vg-black">
            {ranked.slice(0, LIST).map((f, i) => {
              const a = easeOut(seg(t, P.list[0] + i * 0.022, P.list[0] + i * 0.022 + 0.06));
              const hot = i < TOP;
              return (
                <li key={f.key} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 border-b border-vg-rule py-2.5" style={{ opacity: Math.min(1, a * 2) }}>
                  <span className="text-[15px] font-semibold">{f.label}</span>
                  <span className={"text-[15px] font-bold tabular-nums " + "text-vg-black"}>
                    <span aria-hidden className="mr-1.5 text-[11px]">{glyph(f.z)}</span>{sign(f.z)}{Math.abs(f.z).toFixed(1)}σ
                  </span>
                  <span className="col-span-2 mt-1.5 block h-[10px] bg-vg-paper">
                    <span className={"block h-full " + (hot ? "bg-vg-black" : "bg-vg-grey")} style={{ width: `${(Math.abs(f.z) / maxZ) * 100 * a}%` }} />
                  </span>
                </li>
              );
            })}
          </ol>
          <p className="mt-3 text-[13px] leading-relaxed text-vg-mute" style={{ opacity: seg(t, P.list[1] - 0.04, P.list[1]) }}>
            30-day change relative to typical 30-day moves; black marks the four largest. Nikkei is left out: this record has no standardized score for it.
          </p>
        </div>
      </div>

      {/* narrative */}
      <div className="mt-10">
        <p className="text-[15px] font-semibold">Risk memo, written by a language model from these moves</p>
        <p className="mt-1 text-[13px] leading-relaxed text-vg-mute">
          Machine-written from a summary of the factor moves above; not an analyst&rsquo;s note.
        </p>
        <blockquote className="relative mt-3 bg-vg-black py-6 pl-8 pr-6 text-[18px] leading-relaxed text-vg-white sm:text-[20px]">
          <span aria-hidden className="absolute inset-y-0 left-0 w-[10px] bg-vg-red" />
          <Typed text={view.text} p={seg(t, ...P.text)} />
        </blockquote>

        <div className="mt-6 flex items-center justify-between gap-4 border-t-[3px] border-vg-black pt-4">
          <p className="text-[13px] text-vg-mute">Saved output from one run of the Scenario &rarr; Narrative workbench; the demo generates live.</p>
          <ReplayButton onClick={replay} />
        </div>
      </div>
    </>
  );
}
