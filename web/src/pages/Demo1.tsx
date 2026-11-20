import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { DemoHeader } from "@/components/DemoHeader";
import { ControlStep, GenerateButton } from "@/components/DemoControls";
import { HatchDefs, hatch } from "@/components/vg/Hatch";
import { ScenarioFan } from "@/components/ScenarioFan";
import { ScenarioComparisonTable } from "@/components/ScenarioComparisonTable";
import { NARRATIVE_FAMILIES, FACTORS, DEFAULT_NARRATIVE, type Length } from "@/data/narratives";
import {
  presentRelatedPeriods,
  scenarioInputsChanged,
  type ScenarioInputs,
} from "@/demo1Presentation";
import { generateScenario, fetchStartDates, fetchContext, type LiveScenario, type Series } from "@/lib/backend";

const LENGTHS: Length[] = ["short", "medium", "full"];

// per-factor display formatting (matches the original terminal table)
function meta(m: string) {
  if (/^IV/.test(m)) return { s: 100, u: "%", d: 1 };
  if (["US2Y", "US10Y", "AAA_OAS", "BBB_OAS"].includes(m)) return { s: 1, u: "%", d: 2 };
  if (m === "SPX") return { s: 1, u: "", d: 1 };
  if (["GOLD", "CRUDE_OIL"].includes(m)) return { s: 1, u: "", d: 2 };
  return { s: 1, u: "", d: 2 }; // USDJPY / DXY / VIX
}
function scale(s: Series, k: number): Series {
  const m = (a: number[]) => (a ?? []).map((v) => v * k);
  return { p10: m(s.p10), p50: m(s.p50), p90: m(s.p90), mean: m(s.mean), paths: (s.paths ?? []).map(m) };
}

const WRAP = "mx-auto max-w-7xl px-4 sm:px-8";
// Typical wall-clock time of one live run, measured against the live service.
const RUN_HINT = "20 seconds";
// After this long without a response, say so (the request keeps running; nothing is retried automatically).
const SLOW_AFTER_S = 90;
// Default start: the landing story's example date (20 Jan 2016), else the most recent available date.
const DEFAULT_START = "2016-01-20";

// Day-30 P10–P90 width of the narrative-conditioned fan relative to the start-only fan.
// A width (spread) readout from the live result, never a direction claim.
function widthRatio(cond: Series, base: Series): number | null {
  const last = cond.p50.length - 1;
  if (last < 0) return null;
  const wc = cond.p90[last] - cond.p10[last];
  const wb = base.p90?.[last] - base.p10?.[last];
  if (!Number.isFinite(wc) || !Number.isFinite(wb) || Math.abs(wb) < 1e-12) return null;
  return Math.abs(wc) / Math.abs(wb);
}

// Text alternative for the fan chart, built from the live result (day-30 P10–P90 with and without words).
function describeFan(
  label: string,
  date: string | undefined,
  cond: Series,
  base: Series,
  fmt: { u: string; d: number },
  ratio: number | null,
): string {
  const last = cond.p50.length - 1;
  const v = (x: number | undefined) => (x == null || !Number.isFinite(x) ? "n/a" : `${x.toFixed(fmt.d)}${fmt.u}`);
  return [
    `${label}: 30-day distribution${date ? ` from ${date}` : ""}.`,
    `Day 30 P10 to P90 with your words: ${v(cond.p10[last])} to ${v(cond.p90[last])}, median ${v(cond.p50[last])}.`,
    `Market state only: ${v(base.p10?.[last])} to ${v(base.p90?.[last])}.`,
    ratio != null ? `Width ratio ${ratio.toFixed(2)}.` : "",
  ].join(" ").trim();
}

function KeyNumber({ label, value, note, last = false }: { label: string; value: string; note: string; last?: boolean }) {
  return (
    <div className={"flex flex-col justify-between gap-2 border-b-[3px] border-vg-black px-0 py-6 sm:py-9 md:border-b-0 md:border-l-[3px] md:px-8 " + (last ? "border-b-0" : "")}>
      <p className="text-[13px] font-bold uppercase tracking-[0.08em] text-vg-mute">{label}</p>
      <p className="text-vg-h1 tabular-nums">{value}</p>
      <p className="text-[14px] leading-snug text-vg-mute">{note}</p>
    </div>
  );
}

function Swatch({ id, tone }: { id: string; tone: "red" | "grey" }) {
  return (
    <svg width="22" height="14" aria-hidden="true" className="shrink-0">
      <HatchDefs id={id} scale={0.6} />
      <rect x="0" y="0" width="22" height="14" fill={hatch(id, tone)} />
    </svg>
  );
}

function LineKey({ cls }: { cls: string }) {
  return <span aria-hidden="true" className={`inline-block w-6 shrink-0 ${cls}`} />;
}

type Stage = "model" | "history";

// The two client-visible steps of a live run, in the order generate() goes through them.
const STAGES: { key: Stage; title: string; detail: string }[] = [
  {
    key: "model",
    title: "Live model run",
    detail: "One call to the ScenarioFlow service: your words are matched to related historical periods, then 30-day paths are sampled for 10 markets and 5 points on the S&P 500 implied-volatility surface, with and without the words.",
  },
  {
    key: "history",
    title: "Market history",
    detail: "Loading the observed market days before the start date for the chart.",
  },
];

// Elapsed seconds since `since` (null = not running), ticking 4× a second.
function useElapsed(since: number | null) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since == null) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [since]);
  return since == null ? 0 : Math.max(0, (now - since) / 1000);
}

function StageList({ stage, dark = false }: { stage: Stage; dark?: boolean }) {
  const at = STAGES.findIndex((s) => s.key === stage);
  return (
    <ol className={"border-t-[3px] " + (dark ? "border-vg-white" : "border-vg-black")}>
      {STAGES.map((s, i) => {
        const state = i < at ? "done" : i === at ? "active" : "pending";
        return (
          <li key={s.key} className={"grid grid-cols-[2rem_minmax(0,1fr)_auto] gap-x-3 border-b py-3 " + (dark ? "border-vg-mute" : "border-vg-rule")}>
            <span className={"text-[22px] font-extrabold leading-none tabular-nums " + (state === "active" ? "text-vg-red" : state === "done" ? (dark ? "text-vg-white" : "text-vg-black") : "text-vg-grey")}>{i + 1}</span>
            <span className="min-w-0">
              <span className={"block text-[15px] font-extrabold " + (state === "pending" ? "text-vg-grey" : "")}>{s.title}</span>
              <span className={"mt-1 block text-[13px] leading-snug " + (dark ? "text-vg-soft" : "text-vg-mute")}>{s.detail}</span>
            </span>
            <span className={"text-[12px] font-bold uppercase tracking-[0.06em] " + (state === "active" ? "text-vg-red" : dark ? "text-vg-soft" : "text-vg-mute")}>
              {state === "done" ? "Done" : state === "active" ? "Running" : ""}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

// Abstract moving 45° hatch for the wait state: decoration only, carries no data.
function HatchRunner({ reduced }: { reduced: boolean }) {
  return (
    <div aria-hidden="true" className="relative h-3 w-full overflow-hidden bg-vg-rule">
      <motion.div
        className="absolute inset-y-0 -left-6 right-0"
        style={{ backgroundImage: `repeating-linear-gradient(45deg, var(--vg-red) 0 3px, transparent 3px 9px)` }}
        animate={reduced ? undefined : { x: [0, 12.7] }}
        transition={reduced ? undefined : { duration: 0.6, ease: "linear", repeat: Infinity }}
      />
    </div>
  );
}

const SELECT_CLS = "h-12 w-full appearance-none rounded-none border-2 border-vg-black bg-vg-white pl-3 pr-10 text-[15px] font-medium text-vg-black outline-none focus-visible:border-vg-red";

function Chevron() {
  return (
    <svg aria-hidden="true" width="12" height="8" viewBox="0 0 12 8" className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
      <path d="M1 1l5 5 5-5" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="square" />
    </svg>
  );
}

export default function Demo1() {
  const shouldReduceMotion = useReducedMotion();
  const chartRef = useRef<HTMLDivElement>(null);
  const generationId = useRef(0);
  const [dates, setDates] = useState<{ date: string; index: number }[]>([]);
  const [idx, setIdx] = useState<number | null>(null);
  const [narrative, setNarrative] = useState(DEFAULT_NARRATIVE);
  const [famKey, setFamKey] = useState(NARRATIVE_FAMILIES[0].key);
  const [len, setLen] = useState<Length>("short");
  const [factor, setFactor] = useState("SPX");
  const [showBase, setShowBase] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LiveScenario | null>(null);
  const [resultRevision, setResultRevision] = useState(0);
  const [submittedInputs, setSubmittedInputs] = useState<(ScenarioInputs & { date?: string }) | null>(null);
  const [context, setContext] = useState<Record<string, number[]> | null>(null);
  const [stage, setStage] = useState<Stage>("model");
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const elapsed = useElapsed(generating ? startedAt : null);

  const [datesLoading, setDatesLoading] = useState(true);
  const [dateLoadError, setDateLoadError] = useState(false);
  const dateLoadId = useRef(0);
  const errorRef = useRef<HTMLDivElement>(null);

  const loadDates = useCallback(async () => {
    const requestId = ++dateLoadId.current;
    setDatesLoading(true);
    setDateLoadError(false);
    try {
      const ds = await fetchStartDates();
      if (requestId !== dateLoadId.current) return;
      if (!ds.length) throw new Error("No starting dates returned");
      setDates(ds);
      setIdx((current) => current != null && ds.some((d) => d.index === current)
        ? current
        : (ds.find((d) => d.date === DEFAULT_START) ?? ds[ds.length - 1]).index);
    } catch (e) {
      if (requestId !== dateLoadId.current) return;
      console.error("[live] fetchStartDates failed:", e);
      setDateLoadError(true);
    } finally {
      if (requestId === dateLoadId.current) setDatesLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void loadDates(), 0);
    return () => {
      window.clearTimeout(initialLoad);
      dateLoadId.current += 1;
    };
  }, [loadDates]);

  function pickExample(fk: string, l: Length) {
    setFamKey(fk); setLen(l);
    const fam = NARRATIVE_FAMILIES.find((f) => f.key === fk);
    if (fam) setNarrative(fam.variants[l]);
  }

  async function generate() {
    if (idx == null || !narrative.trim()) return;
    const requestId = ++generationId.current;
    const request = {
      windowIndex: idx,
      narrative: narrative.trim(),
      date: dates.find((date) => date.index === idx)?.date,
    };
    setGenerating(true); setError(null);
    setStage("model"); setStartedAt(Date.now());
    // on single-column layouts the chart sits below the controls: bring the live status into view
    if (window.innerWidth < 1024) {
      requestAnimationFrame(() => chartRef.current?.scrollIntoView({ behavior: shouldReduceMotion ? "auto" : "smooth", block: "start" }));
    }
    try {
      const res = await generateScenario(request.narrative, request.windowIndex);
      if (requestId === generationId.current) setStage("history");
      const nextContext = await fetchContext(request.windowIndex).catch(() => null);
      if (requestId !== generationId.current) return;
      setResult(res);
      setContext(nextContext);
      setSubmittedInputs(request);
      setResultRevision((revision) => revision + 1);
    } catch (e) {
      console.error("[live] generateScenario failed:", e);
      if (requestId !== generationId.current) return;
      setError(String((e as Error)?.message || e));
      // on single-column layouts the chart (and its error panel) is in view; elsewhere bring the alert in
      if (window.innerWidth >= 1024) {
        requestAnimationFrame(() => errorRef.current?.scrollIntoView({ behavior: shouldReduceMotion ? "auto" : "smooth", block: "nearest" }));
      }
    } finally {
      if (requestId === generationId.current) setGenerating(false);
    }
  }

  const ff = result?.factors[factor];
  const k = meta(factor).s;
  const displayedDate = submittedInputs?.date;
  const relatedPeriods = result ? presentRelatedPeriods(result.analogues) : [];
  const inputsChanged = idx != null && scenarioInputsChanged(
    { windowIndex: idx, narrative },
    submittedInputs,
  );

  const ratio = ff ? widthRatio(ff.cond, ff.base) : null;
  const factorLabel = FACTORS.find(([, k2]) => k2 === factor)?.[0] ?? factor;
  const fanLabel = ff ? describeFan(factorLabel, displayedDate, scale(ff.cond, k), scale(ff.base, k), meta(factor), ratio) : "";
  const spreadRatios: Record<string, number | null> = {};
  if (result) for (const [key, fan] of Object.entries(result.factors)) spreadRatios[key] = widthRatio(fan.cond, fan.base);
  const ease = [0.65, 0, 0.35, 1] as const;

  function viewDistribution(market: string) {
    setFactor(market);
    requestAnimationFrame(() => {
      chartRef.current?.focus({ preventScroll: true });
      chartRef.current?.scrollIntoView({
        behavior: shouldReduceMotion ? "auto" : "smooth",
        block: "start",
      });
    });
  }
  return (
    <main className="bg-vg-white text-vg-black">
      <DemoHeader
        kicker="Narrative → Scenario"
        line1="Describe the market."
        line2="Get the distribution."
        lede="Write what you see in the market and pick a starting date. ScenarioFlow generates a conditional distribution of 30-day paths across 10 markets and 5 implied-volatility points, live."
        steps={[
          { title: "Your narrative", detail: "Write what you see in the market, or load an example." },
          { title: "Starting date", detail: "The market state the 30-day paths start from." },
          { title: "Market to view", detail: "Compare its distribution with and without your words." },
        ]}
      />

      {/* ---- workbench ---- */}
      <section className={`${WRAP} py-10 sm:py-14`}>
        <div className="grid gap-10 lg:grid-cols-[380px_minmax(0,1fr)] lg:gap-12">
          {/* ---- control panel ---- */}
          <div className="h-fit">
            <ControlStep n={1} label="Your narrative" htmlFor="d1-narr" first>
            <textarea id="d1-narr" value={narrative} onChange={(e) => setNarrative(e.target.value)}
              className="mt-3 min-h-36 w-full rounded-none border-2 border-vg-black bg-vg-white p-3 text-[15px] leading-relaxed outline-none focus-visible:border-vg-red" />
            <div className="mt-4">
              <p id="d1-ex-label" className="text-sm font-semibold text-vg-mute">Or load an example</p>
              <div role="group" aria-labelledby="d1-ex-label" className="mt-2 flex flex-wrap gap-1.5">
                {NARRATIVE_FAMILIES.map((f) => {
                  const on = famKey === f.key;
                  return (
                    <button key={f.key} type="button" aria-pressed={on} onClick={() => pickExample(f.key, len)}
                      className={`relative min-h-9 border-l-[6px] px-2.5 py-1.5 text-left text-[13px] font-semibold leading-tight transition-colors ${on ? "border-vg-red bg-vg-black text-vg-white" : "border-vg-black bg-vg-paper text-vg-black hover:bg-vg-black hover:text-vg-white"}`}>
                      {f.label}
                    </button>
                  );
                })}
              </div>
              <div role="group" aria-label="Example length" className="mt-2 grid grid-cols-3 border-2 border-vg-black">
                {LENGTHS.map((l) => (
                  <button key={l} type="button" aria-pressed={len === l} onClick={() => pickExample(famKey, l)}
                    className={`min-h-9 px-2 text-[13px] font-semibold capitalize transition-colors [&:not(:first-child)]:border-l-2 [&:not(:first-child)]:border-vg-black ${len === l ? "bg-vg-black text-vg-white" : "bg-vg-white text-vg-black hover:bg-vg-paper"}`}>{l}</button>
                ))}
              </div>
            </div>

            </ControlStep>

            <ControlStep n={2} label="Starting date" htmlFor="d1-date">
              <div className="relative mt-3">
                <select id="d1-date" value={idx ?? ""} onChange={(e) => setIdx(Number(e.target.value))}
                  disabled={datesLoading || dateLoadError}
                  className={`${SELECT_CLS} tabular-nums disabled:cursor-not-allowed disabled:border-vg-soft disabled:text-vg-grey`}>
                  {dates.length === 0 && <option value="">{datesLoading ? "Loading dates…" : "Dates unavailable"}</option>}
                  {dates.map((d) => <option key={d.index} value={d.index}>{d.date}</option>)}
                </select>
                <Chevron />
              </div>
              <p className="mt-2 text-[13px] text-vg-mute">Type a year to search. Dates span 2000–2016, including the 2008 crisis.</p>
              {dateLoadError && (
                <div className="mt-4 border-l-[6px] border-vg-red bg-vg-paper p-3" role="alert">
                  <p className="text-[13px] font-semibold text-vg-red">Starting dates could not be loaded.</p>
                  <button type="button" onClick={() => void loadDates()}
                    className="mt-2 text-[13px] font-bold text-vg-black underline decoration-2 underline-offset-4 hover:text-vg-red">
                    Retry loading dates
                  </button>
                </div>
              )}
            </ControlStep>

            <ControlStep n={3} label="Market to view" htmlFor="d1-factor">
              <div className="relative mt-3">
                <select id="d1-factor" value={factor} onChange={(e) => setFactor(e.target.value)} className={SELECT_CLS}>
                  {FACTORS.map(([label, key]) => <option key={key} value={key}>{label}</option>)}
                </select>
                <Chevron />
              </div>
            </ControlStep>

            <GenerateButton
              onClick={generate}
              disabled={generating || idx == null || !narrative.trim()}
              busy={generating}
              label="Generate 30-day scenarios"
              busyLabel={result ? "Updating scenarios…" : "Generating…"}
              elapsed={elapsed}
            />
            <div ref={errorRef} className="scroll-mt-24">
              {error && <p className="mt-3 border-l-[6px] border-vg-red pl-3 text-sm font-semibold text-vg-red" role="alert">Generation failed. Check that the ScenarioFlow service is running, then try again.</p>}
            </div>
            {generating && elapsed >= SLOW_AFTER_S && (
              <p className="mt-3 border-l-[6px] border-vg-black bg-vg-paper py-2 pl-3 pr-2 text-sm font-semibold" role="status">
                Still waiting for the ScenarioFlow service. A live run usually takes about {RUN_HINT}; reload the page to stop waiting.
              </p>
            )}
            {inputsChanged && result && !generating && (
              <p className="mt-3 border-l-[6px] border-vg-red bg-vg-paper py-2 pl-3 pr-2 text-sm font-semibold" role="status">
                Inputs changed. Generate again to update the displayed result.
              </p>
            )}
            <p className="sr-only" aria-live="polite">
              {generating ? (result ? "Updating scenarios" : "Generating scenarios") : result ? "Scenarios ready" : ""}
            </p>
            <p className="mt-4 text-[13px] leading-relaxed text-vg-mute">Related historical periods shape the distribution.</p>
          </div>

          {/* ---- chart ---- */}
          <div className="relative min-w-0" aria-busy={generating}>
            <motion.div
              className={generating && result ? "pointer-events-none" : ""}
              animate={{ opacity: generating && result ? 0.4 : 1 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            >
              <div ref={chartRef} data-testid="d1-chart" tabIndex={-1} className="scroll-mt-20 border-t-8 border-vg-black pt-5 outline-none focus-visible:outline-3 focus-visible:outline-offset-4 focus-visible:outline-vg-red">
                <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
                  <div className="min-w-0">
                    <h2 className="text-vg-h2">{factorLabel}</h2>
                    <p className="mt-1 text-[15px] text-vg-mute">
                      30-day distribution{displayedDate ? ` from ${displayedDate}` : ""}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    {result && <Badge variant="secondary" className="h-8 px-3 text-[13px]">{result.support}</Badge>}
                    <button
                      type="button"
                      aria-pressed={showBase}
                      onClick={() => setShowBase((visible) => !visible)}
                      className="inline-flex min-h-11 items-center gap-2.5 border-2 border-vg-black bg-vg-white px-3 text-[13px] font-semibold transition-colors hover:bg-vg-paper"
                    >
                      <span className={`relative h-5 w-9 border-2 border-vg-black transition-colors ${showBase ? "bg-vg-black" : "bg-vg-white"}`} aria-hidden="true">
                        <motion.span
                          className={`absolute left-0.5 top-0.5 h-3 w-3 ${showBase ? "bg-vg-red" : "bg-vg-black"}`}
                          animate={{ x: showBase ? 16 : 0 }}
                          transition={{ duration: shouldReduceMotion ? 0 : 0.18, ease: "easeOut" }}
                        />
                      </span>
                      Compare start-only
                    </button>
                  </div>
                </div>

                <ul className="mt-5 flex flex-wrap gap-x-5 gap-y-2 border-b-2 border-vg-black pb-3 text-[13px] font-medium" aria-label="Chart key">
                  <li className="flex items-center gap-2"><LineKey cls="h-1 bg-vg-black" />Market so far</li>
                  <li className="flex items-center gap-2"><Swatch id="d1-lg-red" tone="red" />+ your words, P10–P90</li>
                  <li className="flex items-center gap-2"><LineKey cls="h-1 bg-vg-red" />Median</li>
                  <li className="flex items-center gap-2"><LineKey cls="border-t-2 border-dotted border-vg-black" />Mean</li>
                  {showBase && <li className="flex items-center gap-2 text-vg-mute"><Swatch id="d1-lg-grey" tone="grey" />Market state only</li>}
                  {showBase && <li className="flex items-center gap-2 text-vg-mute"><LineKey cls="border-t-2 border-dashed border-vg-mute" />Start-only median</li>}
                </ul>

                <AnimatePresence mode="wait" initial={false}>
                  {ff ? (
                    <motion.div
                      key={`${resultRevision}-${factor}`}
                      className="mt-2"
                      initial={shouldReduceMotion ? { opacity: 1 } : { clipPath: "inset(0 100% 0 0)" }}
                      animate={shouldReduceMotion ? { opacity: 1 } : { clipPath: "inset(0 0% 0 0)" }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: shouldReduceMotion ? 0 : 0.7, ease }}
                    >
                      <div role="img" aria-label={fanLabel}>
                        <ScenarioFan cond={scale(ff.cond, k)} base={scale(ff.base, k)} showBase={showBase}
                          history={context?.[factor]?.map((v) => v * k)} unit={meta(factor).u} height={400} />
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div key={generating ? "loading" : "empty"} className="mt-2 min-h-[400px]"
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}>
                      {generating ? (
                        <div className="grid min-h-[400px] bg-vg-black text-vg-white sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                          <div className="flex flex-col justify-between gap-6 border-b-[3px] border-vg-red p-6 sm:border-b-0 sm:border-r-[3px] sm:p-8">
                            <p className="text-[15px] font-semibold text-vg-soft">Live generation · elapsed</p>
                            <p className="text-vg-mega tabular-nums" aria-hidden="true">
                              {Math.floor(elapsed)}<span className="text-[0.4em] tracking-[-0.02em] text-vg-red"> s</span>
                            </p>
                            <HatchRunner reduced={!!shouldReduceMotion} />
                          </div>
                          <div className="flex flex-col justify-center gap-5 p-6 sm:p-8">
                            <p className="text-[24px] font-extrabold leading-tight tracking-[-0.03em]">Generating scenarios</p>
                            <StageList stage={stage} dark />
                            <p className="text-[13px] text-vg-soft">A live run usually takes about {RUN_HINT}. Nothing is cached: every result is generated now.</p>
                          </div>
                        </div>
                      ) : (
                        <div className="grid min-h-[400px] border-[3px] border-vg-black sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                          <div className="flex flex-col justify-center gap-4 p-6 sm:p-10">
                            {error ? (
                              <div role="alert" className="border-l-[6px] border-vg-red pl-4">
                                <p className="text-[32px] font-extrabold leading-tight tracking-[-0.03em]">Generation failed.</p>
                                <p className="mt-3 max-w-md text-[15px] text-vg-mute">The ScenarioFlow service did not return a scenario. Check that it is running, then generate again.</p>
                              </div>
                            ) : (
                              <>
                                <p className="text-[32px] font-extrabold leading-tight tracking-[-0.03em]">No scenario yet.</p>
                                <p className="max-w-md text-[15px] text-vg-mute">Write a narrative, pick a starting date and generate. The red fan is the distribution with your words; switch on the comparison to see the grey fan without them.</p>
                              </>
                            )}
                          </div>
                          <div aria-hidden="true" className="relative hidden border-l-[3px] border-vg-black sm:block">
                            <svg className="absolute inset-0 h-full w-full">
                              <HatchDefs id="d1-empty" />
                              <rect x="0" y="0" width="50%" height="100%" fill={hatch("d1-empty", "grey")} />
                              <rect x="50%" y="0" width="50%" height="100%" fill={hatch("d1-empty", "red")} />
                            </svg>
                            <span className="absolute bottom-3 left-3 bg-vg-white px-2 py-1 text-[12px] font-bold text-vg-mute">Market state only</span>
                            <span className="absolute bottom-3 right-3 bg-vg-white px-2 py-1 text-[12px] font-bold text-vg-red">+ your words</span>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
                <p className="mt-3 text-sm leading-relaxed text-vg-mute">
                  Red shows the distribution generated with language. Turn on <em>Compare start-only</em> to overlay the distribution from the same starting date without language.
                </p>
              </div>
            </motion.div>

            {generating && result && (
              <div className="pointer-events-none absolute inset-x-0 top-40 z-10 flex justify-center px-2" role="status" aria-live="polite">
                <div className="w-full max-w-md border-l-[10px] border-vg-red bg-vg-black p-5 text-vg-white">
                  <p className="flex items-baseline justify-between gap-4">
                    <span className="text-[19px] font-extrabold tracking-[-0.02em]">Updating scenarios</span>
                    <span className="text-[28px] font-extrabold tabular-nums tracking-[-0.04em]">{Math.floor(elapsed)}<span className="text-[15px] text-vg-red"> s</span></span>
                  </p>
                  <div className="mt-3"><StageList stage={stage} dark /></div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <AnimatePresence mode="wait" initial={false}>
        {result && (
          <motion.div
            key={resultRevision}
            className={generating ? "pointer-events-none" : ""}
            initial={{ opacity: 0 }}
            animate={{ opacity: generating ? 0.45 : 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.24, ease: "easeOut" }}
          >
            {/* ---- readout: key numbers of this live result ---- */}
            <section className="border-y-[3px] border-vg-black" aria-label="Key numbers of this result">
              <div className={`${WRAP} grid gap-0 md:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)_minmax(0,1fr)]`}>
                <div className="bg-vg-red px-5 py-7 text-vg-white sm:px-8 sm:py-9">
                  <p className="text-[13px] font-bold uppercase tracking-[0.08em] text-vg-white/80">Day-30 spread · {factorLabel}</p>
                  {ratio != null ? (
                    <motion.p
                      key={`${resultRevision}-${factor}`}
                      className="mt-2 text-vg-mega tabular-nums"
                      initial={shouldReduceMotion ? false : { clipPath: "inset(0 100% 0 0)" }}
                      animate={{ clipPath: "inset(0 0% 0 0)" }}
                      transition={{ duration: shouldReduceMotion ? 0 : 0.6, delay: shouldReduceMotion ? 0 : 0.3, ease }}
                    >
                      ×{ratio.toFixed(2)} <span className="text-[0.4em] tracking-[-0.02em]">{ratio >= 1 ? "wider" : "narrower"}</span>
                    </motion.p>
                  ) : <p className="mt-2 text-vg-mega">—</p>}
                  <p className="mt-3 max-w-md text-[15px] leading-snug">
                    P10–P90 width with your words, relative to market state only. ×1.00 means the words left the spread unchanged;
                    the ratio describes how wide the outcomes are, not which way they go.
                  </p>
                </div>
                <KeyNumber label="Related periods" value={String(relatedPeriods.length)} note="historical periods retrieved for your words" />
                <KeyNumber
                  label="Closest match"
                  value={relatedPeriods.length ? Math.max(...relatedPeriods.map((r) => r.matchPercent / 100)).toFixed(2) : "—"}
                  note="narrative similarity of the best-matching period"
                  last
                />
              </div>
            </section>

            {/* ---- related historical periods (black band) ---- */}
            <section className="bg-vg-black text-vg-white" data-testid="d1-result">
              <div className={`${WRAP} py-12 sm:py-16`}>
                <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:items-end lg:gap-10">
                  <h2 className="text-vg-h2">Related historical periods</h2>
                  <p className="text-[15px] leading-relaxed text-vg-soft">
                    Dates are the last day of each matched 30-day window. The bar shows narrative similarity. Path share shows how many scenarios use that period.
                    Periods are matched across the whole 2000–2016 library, including periods after the starting date:
                    this is a what-if tool, not a backtest.
                  </p>
                </div>
                <ul className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {relatedPeriods.map((period, i) => (
                    <li key={period.label} className="relative bg-vg-panel py-4 pl-6 pr-4">
                      <span aria-hidden="true" className="absolute inset-y-0 left-0 w-2.5 bg-vg-red" />
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="text-[26px] font-extrabold leading-none tracking-[-0.03em] tabular-nums">{period.label}</span>
                        <span className="text-sm font-semibold tabular-nums text-vg-soft">match {period.matchLabel}</span>
                      </div>
                      <span className="mt-4 block h-1.5 bg-vg-mute">
                        <motion.span
                          className="block h-full bg-vg-red"
                          initial={{ width: shouldReduceMotion ? `${period.matchPercent}%` : 0 }}
                          whileInView={{ width: `${period.matchPercent}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: shouldReduceMotion ? 0 : 0.5, delay: shouldReduceMotion ? 0 : i * 0.06, ease }}
                        />
                      </span>
                      <span className="mt-2 block text-[13px] text-vg-soft">{period.pathShareLabel}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            {/* ---- comparison table ---- */}
            <section className={`${WRAP} pb-16 pt-12 sm:pt-16`} aria-label="Generated scenario results">
              <ScenarioComparisonTable
                rows={result.comparisonRows}
                startDate={displayedDate}
                selectedMarket={factor}
                onSelectMarket={viewDistribution}
                spreadRatios={spreadRatios}
              />
            </section>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
