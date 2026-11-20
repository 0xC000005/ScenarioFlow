import { HatchDefs, hatch } from "@/components/vg/Hatch";
import { VG } from "@/lib/palette";
import { fmtDate, pts, ratioText, ratioWord, scaleFor, useShift, type ShiftMarket } from "./shift";

// Before and after, all at once: for every market in the 20 Jan 2016 query's output, the 30-day range from market
// state only (grey) and the range once the words are added (red), overlaid, with the two day-30 brackets.
// Static on purpose: nothing has to play for the comparison to be visible.

const W = 320, H = 170;
const GX = 2, GW = 262, GY = 8, GH = H - 16; // right 58 px hold the brackets

function Panel({ m, i }: { m: ShiftMarket; i: number }) {
  const [lo, hi] = scaleFor(m);
  const Hn = m.hist.length;
  const X = (d: number) => GX + (GW * (d + Hn - 1)) / (Hn - 1 + 30);
  const Y = (v: number) => GY + GH * (1 - (v - lo) / (hi - lo));
  const area = (b: ShiftMarket["base"]) =>
    `${pts(b.p90.map((v, d) => [X(d), Y(v)]))} ${pts(b.p10.map((v, d) => [X(d), Y(v)]).reverse())}`;
  const edge = (arr: number[]) => pts(arr.map((v, d) => [X(d), Y(v)]));
  const id = `ba${i}`;
  const bracket = (x: number, y0: number, y1: number, c: string) => (
    <g stroke={c} strokeWidth={3.5}>
      <line x1={x} y1={y0} x2={x} y2={y1} />
      <line x1={x - 6} y1={y0} x2={x + 6} y2={y0} />
      <line x1={x - 6} y1={y1} x2={x + 6} y2={y1} />
    </g>
  );
  const wider = ratioWord(m.ratio) === "wider";
  const span = (b: ShiftMarket["base"]) => (b.p90[30] - b.p10[30]).toPrecision(3);
  return (
    <figure className="border-t-4 border-vg-black pt-2">
      <figcaption className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-2">
        <span className="text-[15px] font-bold leading-tight sm:text-[17px]">{m.name}</span>
        <span className={"whitespace-nowrap text-[15px] font-extrabold tabular-nums sm:text-[18px] " + (wider ? "text-vg-red" : "text-vg-black")}>
          {ratioText(m.ratio)}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 block h-auto w-full" role="img"
        aria-label={`${m.name}: day-30 10–90% range from ${span(m.base)} (market state only) to ${span(m.cond)} with the words, ${ratioText(m.ratio, 2)}`}>
        <HatchDefs id={id} scale={0.6} />
        <HatchDefs id={id + "b"} scale={1.1} />
        <line x1={X(0)} y1={GY - 4} x2={X(0)} y2={GY + GH + 4} stroke={VG.rule} strokeWidth={2} />
        {/* red (with the words) underneath, lighter; the grey "before" band on top in a coarser hatch with heavier
            edges, so both envelopes read as bands whether the words widen or narrow the range */}
        <polygon points={area(m.cond)} fill={hatch(id, "red")} opacity={0.5} />
        <polygon points={area(m.base)} fill={hatch(id + "b", "white")} opacity={0.9} />
        <polyline fill="none" stroke={VG.red} strokeWidth={2.4} points={edge(m.cond.p90)} />
        <polyline fill="none" stroke={VG.red} strokeWidth={2.4} points={edge(m.cond.p10)} />
        <polyline fill="none" stroke={VG.grey} strokeWidth={2.6} strokeDasharray="7 4" points={edge(m.base.p90)} />
        <polyline fill="none" stroke={VG.grey} strokeWidth={2.6} strokeDasharray="7 4" points={edge(m.base.p10)} />
        <polyline fill="none" stroke={VG.black} strokeWidth={2.6} strokeLinejoin="round" points={pts(m.hist.map((v, j) => [X(j - Hn + 1), Y(v)]))} />
        {bracket(X(30) + 18, Y(m.base.p90[30]), Y(m.base.p10[30]), VG.grey)}
        {bracket(X(30) + 42, Y(m.cond.p90[30]), Y(m.cond.p10[30]), VG.red)}
      </svg>
    </figure>
  );
}

export default function BeforeAfterGrid() {
  const { d, failed } = useShift();
  if (failed) return <p className="text-[15px] text-vg-mute">The model output could not load.</p>;
  if (!d) return <div className="min-h-[420px]" />;
  return (
    <div>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 border-b-2 border-vg-black pb-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end lg:gap-10">
        <div className="min-w-0">
          <p className="text-[13px] font-semibold text-vg-red">Your words, 20 Jan 2016</p>
          <p className="mt-1 max-w-3xl text-[16px] italic leading-snug">&ldquo;{d.story}&rdquo;</p>
          <p className="mt-3 text-[14px] text-vg-mute">
            Past markets the model drew on for these words:{" "}
            {d.precedents.map((p, i) => (
              <span key={p.date}>
                {i > 0 ? ", " : ""}
                <span className="whitespace-nowrap font-semibold text-vg-black">
                  {fmtDate(p.date)} <span className="font-normal text-vg-mute">(match {p.cosine.toFixed(2)})</span>
                </span>
              </span>
            ))}
          </p>
        </div>
        <ul className="flex flex-wrap gap-x-6 gap-y-2 text-[15px] font-semibold" aria-label="Legend">
          <li className="inline-flex items-center gap-2 text-vg-mute">
            <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden>
              <HatchDefs id="ba-lg" scale={0.75} />
              <HatchDefs id="ba-lgb" scale={1.1} />
              <rect x="1" y="1" width="28" height="14" fill={hatch("ba-lgb", "white")} stroke={VG.grey} strokeWidth="2" strokeDasharray="7 4" />
            </svg>
            market state only
          </li>
          <li className="inline-flex items-center gap-2 text-vg-red">
            <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden>
              <rect x="1" y="1" width="28" height="14" fill={hatch("ba-lg", "red")} stroke={VG.red} strokeWidth="2" />
            </svg>
            + your words
          </li>
        </ul>
      </div>
      <div className="mt-6 grid grid-cols-[repeat(2,minmax(0,1fr))] gap-x-4 gap-y-6 sm:gap-x-6 lg:grid-cols-[repeat(4,minmax(0,1fr))] lg:gap-x-8 lg:gap-y-8">
        {d.markets.map((m, i) => (
          <Panel key={m.key} m={m} i={i} />
        ))}
      </div>
      <p className="mt-6 max-w-3xl text-[14px] leading-relaxed text-vg-mute">
        Brackets: each market&rsquo;s 10–90% range on day 30, grey without the words and red with them.
      </p>
    </div>
  );
}
