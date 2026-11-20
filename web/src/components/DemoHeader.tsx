import type { ReactNode } from "react";

/**
 * Shared header band for both demo pages: black band, red bar + direction kicker ("Narrative → Scenario"), a two-line
 * headline at one scale (second line red), a lede, and a numbered "how it works" column whose
 * numbers and titles match the numbered sections of the page below.
 */
export function DemoHeader({
  kicker,
  line1,
  line2,
  lede,
  steps,
}: {
  kicker: string;
  line1: string;
  line2: string;
  lede: ReactNode;
  steps: { title: string; detail: string }[];
}) {
  return (
    <section className="relative overflow-hidden bg-vg-black text-vg-white">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 pb-12 pt-12 sm:px-8 sm:pb-16 sm:pt-16 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-end lg:gap-16">
        <div className="min-w-0">
          <p className="flex items-center gap-3 text-[15px] font-semibold text-vg-soft">
            <span aria-hidden="true" className="block h-5 w-2.5 bg-vg-red" />
            {kicker}
          </p>
          <h1 className="mt-6 text-vg-h1">
            {line1}
            <span className="block text-vg-red">{line2}</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-vg-soft">{lede}</p>
        </div>
        <ol className="border-t-[3px] border-vg-white text-[15px]" aria-label="How this demo works">
          {steps.map((step, i) => (
            <li key={step.title} className="grid grid-cols-[2.25rem_minmax(0,1fr)] items-baseline border-b border-vg-mute py-3.5">
              <span className="text-[24px] font-extrabold leading-none tracking-[-0.04em] text-vg-red tabular-nums">{i + 1}</span>
              <span>
                <span className="block font-extrabold text-vg-white">{step.title}</span>
                <span className="mt-0.5 block text-[14px] leading-snug text-vg-soft">{step.detail}</span>
              </span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
