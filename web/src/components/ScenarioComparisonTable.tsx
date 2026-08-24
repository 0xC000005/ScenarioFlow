import { Card } from "@/components/ui/card";
import {
  PAPER_TONE_COLORS,
  comparisonPresentation,
  directionPresentation,
  type DirectionLabel,
  type DistributionReadout,
  type ScenarioComparisonRow,
  type Tone,
} from "@/scenarioComparison";

function Direction({ value }: { value: DirectionLabel }) {
  const presentation = directionPresentation(value);
  if (value === "Mixed" || value === "n/a") {
    return <span style={{ color: PAPER_TONE_COLORS.gray }}>{value}</span>;
  }
  return (
    <span className="whitespace-nowrap">
      <span className="font-semibold" style={{ color: PAPER_TONE_COLORS[presentation.tone] }}>
        {presentation.arrow}
      </span>{" "}
      {value}
    </span>
  );
}

function Comparison({ value }: { value: string }) {
  const presentation = comparisonPresentation(value);
  return (
    <span className="inline-flex items-start gap-1.5">
      {presentation.arrow && (
        <span
          aria-hidden="true"
          className="shrink-0 font-semibold"
          style={{ color: PAPER_TONE_COLORS[presentation.tone] }}
        >
          {presentation.arrow}
        </span>
      )}
      <span>{value}</span>
    </span>
  );
}

function Move({ value, stacked = false }: { value: string; stacked?: boolean }) {
  const [nativeMove, sigmaMove] = value.split(" / ");
  if (!stacked || !sigmaMove) return <span className="whitespace-nowrap">{value}</span>;
  return (
    <span>
      <span className="block">{nativeMove}</span>
      <span className="block text-muted-foreground">{sigmaMove}</span>
    </span>
  );
}

function Distribution({ value }: { value: DistributionReadout }) {
  return (
    <dl className="grid grid-cols-3 gap-2 text-sm">
      <div>
        <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Direction</dt>
        <dd className="mt-1"><Direction value={value.direction} /></dd>
      </div>
      <div>
        <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Majority</dt>
        <dd className="mt-1 tabular-nums">{value.majorityShare}</dd>
      </div>
      <div>
        <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Mean move</dt>
        <dd className="mt-1 tabular-nums"><Move value={value.meanMove} stacked /></dd>
      </div>
    </dl>
  );
}

function toneBorder(tone: Tone): string {
  return PAPER_TONE_COLORS[tone];
}

export function ScenarioComparisonTable({
  rows,
  startDate,
  selectedMarket,
  onSelectMarket,
}: {
  rows: ScenarioComparisonRow[];
  startDate?: string;
  selectedMarket?: string;
  onSelectMarket?: (market: string) => void;
}) {
  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border px-5 py-4 sm:px-6">
        <h2 className="text-base font-semibold">Day-30 market comparison</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Start-only and narrative-conditioned distributions{startDate ? ` from ${startDate}` : ""}.
        </p>
      </div>

      <div className="hidden overflow-x-auto lg:block">
        <table className="w-full min-w-[980px] border-collapse text-[13px] leading-snug tabular-nums">
          <caption className="sr-only">
            Start-only and narrative-conditioned day-30 market distributions.
          </caption>
          <colgroup>
            <col className="w-[10%]" />
            <col className="w-[8%]" />
            <col className="w-[10%]" />
            <col className="w-[14%]" />
            <col className="w-[8%]" />
            <col className="w-[10%]" />
            <col className="w-[14%]" />
            <col className="w-[26%]" />
          </colgroup>
          <thead>
            <tr className="border-b border-[#b8b8bd] text-left">
              <th rowSpan={2} scope="col" className="px-4 py-3 align-bottom font-semibold">Market</th>
              <th colSpan={3} scope="colgroup" className="bg-[#f7f7f8] px-3 py-2 text-center font-semibold">
                Start-only distribution
              </th>
              <th colSpan={3} scope="colgroup" className="bg-[#f4f8fd] px-3 py-2 text-center font-semibold">
                Narrative-conditioned distribution
              </th>
              <th rowSpan={2} scope="col" className="px-4 py-3 align-bottom font-semibold">
                Narrative vs. start-only
              </th>
            </tr>
            <tr className="border-b border-[#8e8e93] text-left text-xs text-muted-foreground">
              <th scope="col" className="bg-[#f7f7f8] px-3 py-2 font-medium">Direction</th>
              <th scope="col" className="bg-[#f7f7f8] px-3 py-2 font-medium">Majority share</th>
              <th scope="col" className="bg-[#f7f7f8] px-3 py-2 font-medium">Mean move</th>
              <th scope="col" className="bg-[#f4f8fd] px-3 py-2 font-medium">Direction</th>
              <th scope="col" className="bg-[#f4f8fd] px-3 py-2 font-medium">Majority share</th>
              <th scope="col" className="bg-[#f4f8fd] px-3 py-2 font-medium">Mean move</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.marketKey}
                className={`border-b border-border/80 last:border-b-0 ${row.marketKey === selectedMarket ? "bg-primary/[0.045]" : "odd:bg-black/[0.012]"}`}
              >
                <th scope="row" className="px-4 py-2.5 text-left font-medium">
                  <button
                    type="button"
                    className="rounded-sm text-left hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                    onClick={() => onSelectMarket?.(row.marketKey)}
                    aria-label={`View ${row.marketLabel} distribution`}
                  >
                    <span className="block">{row.marketLabel}</span>
                    <span className="mt-0.5 block text-[10px] font-normal text-muted-foreground">View distribution</span>
                  </button>
                </th>
                <td className="bg-[#f7f7f8]/70 px-3 py-2.5"><Direction value={row.startOnly.direction} /></td>
                <td className="bg-[#f7f7f8]/70 px-3 py-2.5 whitespace-nowrap">{row.startOnly.majorityShare}</td>
                <td className="bg-[#f7f7f8]/70 px-3 py-2.5"><Move value={row.startOnly.meanMove} /></td>
                <td className="bg-[#f4f8fd]/65 px-3 py-2.5"><Direction value={row.narrative.direction} /></td>
                <td className="bg-[#f4f8fd]/65 px-3 py-2.5 whitespace-nowrap">{row.narrative.majorityShare}</td>
                <td className="bg-[#f4f8fd]/65 px-3 py-2.5"><Move value={row.narrative.meanMove} /></td>
                <td className="px-4 py-2.5"><Comparison value={row.comparison} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="divide-y divide-border lg:hidden">
        {rows.map((row) => {
          const presentation = comparisonPresentation(row.comparison);
          return (
            <article
              key={row.marketKey}
              className={row.marketKey === selectedMarket ? "bg-primary/[0.035] px-4 py-4" : "px-4 py-4"}
              style={{ borderLeft: `3px solid ${toneBorder(presentation.tone)}` }}
            >
              <div className="flex items-start justify-between gap-4">
                <button
                  type="button"
                  className="text-left font-semibold hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                  onClick={() => onSelectMarket?.(row.marketKey)}
                  aria-label={`View ${row.marketLabel} distribution`}
                >
                  <span className="block">{row.marketLabel}</span>
                  <span className="mt-0.5 block text-[10px] font-normal text-muted-foreground">View distribution</span>
                </button>
                <span className="max-w-[62%] text-right text-xs"><Comparison value={row.comparison} /></span>
              </div>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <section className="rounded-md bg-[#f7f7f8] p-3">
                  <h3 className="mb-2 text-xs font-semibold">Start-only</h3>
                  <Distribution value={row.startOnly} />
                </section>
                <section className="rounded-md bg-[#f4f8fd] p-3">
                  <h3 className="mb-2 text-xs font-semibold">With narrative</h3>
                  <Distribution value={row.narrative} />
                </section>
              </div>
            </article>
          );
        })}
      </div>

      <p className="border-t border-border px-5 py-3 text-xs leading-relaxed text-muted-foreground sm:px-6">
        Direction is Up or Down when at least 60% of paths finish on that side of the starting value; otherwise it is Mixed.
      </p>
    </Card>
  );
}
