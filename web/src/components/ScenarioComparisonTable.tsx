import { VG } from "@/lib/palette";
import {
  comparisonPresentation,
  directionPresentation,
  type DirectionLabel,
  type DistributionReadout,
  type ScenarioComparisonRow,
  type Tone,
} from "@/scenarioComparison";

// Colour semantics: no green/amber. Arrows carry the direction; colour only separates
// a decisive read (black) from a hedged one (mute) or no read at all (grey).
// Red is reserved for "narrative-conditioned" (the key column), never for "down".
const TONE: Record<Tone, string> = {
  green: VG.black,
  red: VG.black,
  amber: VG.mute,
  gray: VG.grey,
};

// Same direction glyphs as the rest of the site (▲/▼); the presentation module's ↑/↓ map onto them.
const glyph = (arrow: string) => (arrow === "↑" ? "▲" : arrow === "↓" ? "▼" : arrow);

// Presentation-only wording: these columns describe where the generated paths finish relative to
// the starting value (a property of the distribution), not a forecast of market direction.
const FINISH: Record<DirectionLabel, string> = { Up: "Mostly above start", Down: "Mostly below start", Mixed: "Mixed", "n/a": "n/a" };
const shareLabel = (share: string) => share.replace(/ up$/, " above").replace(/ down$/, " below");
const COMPARISON: Record<string, string> = {
  "More up than baseline": "Mean higher, further above start",
  "Higher than baseline": "Mean higher",
  "Less down than baseline": "Mean higher, less far below start",
  "More down than baseline": "Mean lower, further below start",
  "Lower than baseline": "Mean lower",
  "Less up than baseline": "Mean lower, less far above start",
  Similar: "Similar mean",
  "Similar to baseline": "Similar mean",
};

function Direction({ value }: { value: DirectionLabel }) {
  const presentation = directionPresentation(value);
  if (value === "Mixed" || value === "n/a") {
    return <span style={{ color: TONE.gray }}>{FINISH[value]}</span>;
  }
  return (
    <span className="whitespace-nowrap">
      <span aria-hidden="true" className="font-extrabold" style={{ color: TONE[presentation.tone] }}>
        <span className="text-[0.8em]">{glyph(presentation.arrow)}</span>
      </span>{" "}
      {FINISH[value]}
    </span>
  );
}

// Day-30 P10–P90 width with the words relative to market state only.
function Spread({ ratio }: { ratio?: number | null }) {
  if (ratio == null || !Number.isFinite(ratio)) return <span className="text-vg-grey">—</span>;
  const r = ratio.toFixed(2);
  const word = r === "1.00" ? "unchanged" : ratio > 1 ? "wider" : "narrower";
  return <span className="whitespace-nowrap font-extrabold">×{r} <span className="text-[0.8em] font-semibold">{word}</span></span>;
}

function Comparison({ value }: { value: string }) {
  const presentation = comparisonPresentation(value);
  return (
    <span className="inline-flex items-start gap-1.5">
      {presentation.arrow && (
        <span
          aria-hidden="true"
          className="shrink-0 font-extrabold"
          style={{ color: TONE[presentation.tone] }}
        >
          <span className="text-[0.8em]">{glyph(presentation.arrow)}</span>
        </span>
      )}
      <span>{COMPARISON[value] ?? value}</span>
    </span>
  );
}

function Move({ value, stacked = false }: { value: string; stacked?: boolean }) {
  const [nativeMove, sigmaMove] = value.split(" / ");
  if (!stacked || !sigmaMove) return <span className="whitespace-nowrap">{value}</span>;
  return (
    <span>
      <span className="block">{nativeMove}</span>
      <span className="block text-vg-mute">{sigmaMove}</span>
    </span>
  );
}

function Distribution({ value }: { value: DistributionReadout }) {
  return (
    <dl className="grid grid-cols-3 gap-2 text-sm">
      <div>
        <dt className="text-xs font-semibold text-vg-mute">Paths finish</dt>
        <dd className="mt-1"><Direction value={value.direction} /></dd>
      </div>
      <div>
        <dt className="text-xs font-semibold text-vg-mute">Share</dt>
        <dd className="mt-1 tabular-nums">{shareLabel(value.majorityShare)}</dd>
      </div>
      <div>
        <dt className="text-xs font-semibold text-vg-mute">Mean move</dt>
        <dd className="mt-1 tabular-nums"><Move value={value.meanMove} stacked /></dd>
      </div>
    </dl>
  );
}

export function ScenarioComparisonTable({
  rows,
  startDate,
  selectedMarket,
  onSelectMarket,
  spreadRatios,
}: {
  rows: ScenarioComparisonRow[];
  startDate?: string;
  selectedMarket?: string;
  onSelectMarket?: (market: string) => void;
  /** Day-30 P10–P90 width ratio (with words / market state only), keyed by market. */
  spreadRatios?: Record<string, number | null>;
}) {
  return (
    <div className="border-t-8 border-vg-black bg-vg-white">
      <div className="grid gap-3 py-6 sm:py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:items-end lg:gap-10">
        <h2 className="text-vg-h2 text-vg-black">Day-30 market comparison</h2>
        <p className="text-[15px] leading-relaxed text-vg-mute">
          Market state only, against market state plus your words{startDate ? `, from ${startDate}` : ""}.
          Select a market to view its distribution.
          <strong className="mt-2 block font-semibold text-vg-black">These columns describe the generated distributions at day 30. They are not a forecast.</strong>
        </p>
      </div>

      <div className="hidden overflow-x-auto lg:block">
        <table className="w-full min-w-[980px] border-collapse text-[14px] leading-snug tabular-nums">
          <caption className="sr-only">
            Start-only and narrative-conditioned day-30 market distributions: spread ratio, where paths finish relative to the start, and mean move. Descriptive of the generated distributions, not a forecast.
          </caption>
          <colgroup>
            <col className="w-[10%]" />
            <col className="w-[8%]" />
            <col className="w-[11%]" />
            <col className="w-[8%]" />
            <col className="w-[11%]" />
            <col className="w-[11%]" />
            <col className="w-[8%]" />
            <col className="w-[11%]" />
            <col className="w-[22%]" />
          </colgroup>
          <thead>
            <tr className="text-left">
              <th rowSpan={2} scope="col" className="border-b-4 border-vg-black px-4 pb-3 pt-4 align-bottom text-[15px] font-extrabold">Market</th>
              <th rowSpan={2} scope="col" className="border-b-4 border-vg-black px-3 pb-3 pt-4 align-bottom text-[15px] font-extrabold">
                Day-30 spread
                <span className="mt-0.5 block text-xs font-semibold text-vg-mute">P10–P90 width, words ÷ state only</span>
              </th>
              <th colSpan={3} scope="colgroup" className="border-t-4 border-vg-grey bg-vg-paper px-3 pb-2 pt-3 text-[15px] font-extrabold text-vg-mute">
                Market state only
              </th>
              <th colSpan={3} scope="colgroup" className="border-l-2 border-t-4 border-vg-red border-l-vg-red px-3 pb-2 pt-3 text-[15px] font-extrabold text-vg-red">
                + your words
              </th>
              <th rowSpan={2} scope="col" className="border-b-4 border-vg-black px-4 pb-3 pt-4 align-bottom text-[15px] font-extrabold">
                Shift of the distribution vs. start-only
              </th>
            </tr>
            <tr className="border-b-4 border-vg-black text-left text-xs text-vg-mute">
              <th scope="col" className="bg-vg-paper px-3 pb-3 pt-1 font-semibold">Where paths finish</th>
              <th scope="col" className="bg-vg-paper px-3 pb-3 pt-1 font-semibold">Share of paths</th>
              <th scope="col" className="bg-vg-paper px-3 pb-3 pt-1 font-semibold">Mean move</th>
              <th scope="col" className="border-l-2 border-vg-red px-3 pb-3 pt-1 font-semibold">Where paths finish</th>
              <th scope="col" className="px-3 pb-3 pt-1 font-semibold">Share of paths</th>
              <th scope="col" className="px-3 pb-3 pt-1 font-semibold">Mean move</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const selected = row.marketKey === selectedMarket;
              return (
                <tr key={row.marketKey} className={`border-b border-vg-rule ${selected ? "bg-vg-paper" : ""}`}>
                  <th scope="row" className="relative px-4 py-3 text-left font-semibold">
                    {selected && <span aria-hidden="true" className="absolute inset-y-0 left-0 w-1.5 bg-vg-red" />}
                    <button
                      type="button"
                      className="group text-left"
                      onClick={() => onSelectMarket?.(row.marketKey)}
                      aria-label={`View ${row.marketLabel} distribution`}
                      aria-current={selected ? "true" : undefined}
                    >
                      <span className="block font-extrabold group-hover:text-vg-red">{row.marketLabel}</span>
                      <span className={`mt-0.5 block text-[11px] font-medium ${selected ? "text-vg-red" : "text-vg-mute"} underline decoration-transparent underline-offset-2 group-hover:decoration-vg-red`}>
                        {selected ? "Showing distribution" : "View distribution"}
                      </span>
                    </button>
                  </th>
                  <td className="px-3 py-3 text-[16px]"><Spread ratio={spreadRatios?.[row.marketKey]} /></td>
                  <td className="bg-vg-paper/70 px-3 py-3 text-vg-mute"><Direction value={row.startOnly.direction} /></td>
                  <td className="bg-vg-paper/70 px-3 py-3 whitespace-nowrap text-vg-mute">{shareLabel(row.startOnly.majorityShare)}</td>
                  <td className="bg-vg-paper/70 px-3 py-3 text-vg-mute"><Move value={row.startOnly.meanMove} /></td>
                  <td className="border-l-2 border-vg-red px-3 py-3 font-semibold"><Direction value={row.narrative.direction} /></td>
                  <td className="px-3 py-3 whitespace-nowrap font-semibold">{shareLabel(row.narrative.majorityShare)}</td>
                  <td className="px-3 py-3 font-semibold"><Move value={row.narrative.meanMove} /></td>
                  <td className="px-4 py-3"><Comparison value={row.comparison} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="border-t-4 border-vg-black lg:hidden">
        {rows.map((row) => {
          const selected = row.marketKey === selectedMarket;
          return (
            <article
              key={row.marketKey}
              className={`relative border-b border-vg-rule py-4 pr-0 ${selected ? "bg-vg-paper pl-4" : "pl-0"}`}
            >
              {selected && <span aria-hidden="true" className="absolute inset-y-0 left-0 w-1.5 bg-vg-red" />}
              <div className={`flex items-start justify-between gap-4 ${selected ? "pr-3" : ""}`}>
                <button
                  type="button"
                  className="group min-h-11 text-left"
                  onClick={() => onSelectMarket?.(row.marketKey)}
                  aria-label={`View ${row.marketLabel} distribution`}
                  aria-current={selected ? "true" : undefined}
                >
                  <span className="block text-lg font-extrabold leading-tight group-hover:text-vg-red">{row.marketLabel}</span>
                  <span className={`mt-0.5 block text-[11px] font-medium ${selected ? "text-vg-red" : "text-vg-mute"}`}>
                    {selected ? "Showing distribution" : "View distribution"}
                  </span>
                </button>
                <span className="max-w-[60%] text-right text-sm font-semibold">
                  <span className="block text-xs font-semibold text-vg-mute">Day-30 spread</span>
                  <span className="block text-[18px]"><Spread ratio={spreadRatios?.[row.marketKey]} /></span>
                  <span className="mt-1 block"><Comparison value={row.comparison} /></span>
                </span>
              </div>
              <div className={`mt-3 grid gap-2 sm:grid-cols-2 ${selected ? "pr-3" : ""}`}>
                <section className="border-t-4 border-vg-grey bg-vg-paper p-3 text-vg-mute">
                  <h3 className="mb-2 text-sm font-extrabold">Market state only</h3>
                  <Distribution value={row.startOnly} />
                </section>
                <section className="border-t-4 border-vg-red bg-vg-white p-3">
                  <h3 className="mb-2 text-sm font-extrabold text-vg-red">+ your words</h3>
                  <Distribution value={row.narrative} />
                </section>
              </div>
            </article>
          );
        })}
      </div>

      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-vg-mute">
        Day-30 spread is the P10–P90 width with your words divided by the width from market state only (×1.00 = unchanged).
        “Mostly above / below start” means at least 60% of generated paths finish on that side of the starting value; otherwise Mixed.
        Your words change the conditional distribution through related historical periods; none of these columns forecast direction.
      </p>
    </div>
  );
}
