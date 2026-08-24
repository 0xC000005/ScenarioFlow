import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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

  useEffect(() => {
    fetchStartDates().then((ds) => {
      setDates(ds);
      if (ds.length) setIdx((ds.find((d) => d.date === "2016-03-24") ?? ds.find((d) => d.date >= "2016-03-24") ?? ds[Math.floor(ds.length / 2)]).index);
    }).catch(() => {});
  }, []);

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
    try {
      const res = await generateScenario(request.narrative, request.windowIndex);
      const nextContext = await fetchContext(request.windowIndex).catch(() => null);
      if (requestId !== generationId.current) return;
      setResult(res);
      setContext(nextContext);
      setSubmittedInputs(request);
      setResultRevision((revision) => revision + 1);
    } catch (e) {
      console.error("[live] generateScenario failed:", e);
      setError(String((e as Error)?.message || e));
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
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">ScenarioFlow · Narrative → Scenario</p>
      </div>
      <h1 className="mt-2 text-display">Describe the market. Generate scenarios.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        Enter a market description and starting date to generate a conditional distribution of 30-day market paths.
      </p>

      <div className="mt-8 grid gap-7 lg:grid-cols-[380px_1fr]">
        {/* ---- input rail ---- */}
        <Card className="h-fit p-6">
          <label htmlFor="d1-narr" className="block text-sm font-medium">Your narrative</label>
          <textarea id="d1-narr" value={narrative} onChange={(e) => setNarrative(e.target.value)}
            className="mt-2 min-h-32 w-full rounded-lg border border-border bg-background p-3 text-[15px] leading-relaxed outline-none focus:ring-2 focus:ring-ring" />

          <div className="mt-3 rounded-lg bg-secondary/60 p-3">
            <div className="text-xs font-medium text-muted-foreground">Or load an example</div>
            <select aria-label="Example regime" value={famKey} onChange={(e) => pickExample(e.target.value, len)}
              className="mt-2 w-full rounded-md border border-border bg-background p-2 text-sm">
              {NARRATIVE_FAMILIES.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
            <div className="mt-2 flex gap-1">
              {LENGTHS.map((l) => (
                <button key={l} onClick={() => pickExample(famKey, l)}
                  className={`flex-1 rounded-md border px-2 py-1 text-xs capitalize transition-colors ${len === l ? "border-primary bg-primary/10 font-medium text-foreground" : "border-border text-muted-foreground hover:bg-secondary"}`}>{l}</button>
              ))}
            </div>
          </div>

          <label htmlFor="d1-date" className="mt-4 block text-sm font-medium">Starting date</label>
          <select id="d1-date" value={idx ?? ""} onChange={(e) => setIdx(Number(e.target.value))}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {dates.length === 0 && <option>loading dates…</option>}
            {dates.map((d) => <option key={d.index} value={d.index}>{d.date}</option>)}
          </select>
          <p className="mt-1 text-xs text-muted-foreground">Type a year to search · spans 2000–2016 incl. the 2008 crisis.</p>

          <label htmlFor="d1-factor" className="mt-4 block text-sm font-medium">Market variable</label>
          <select id="d1-factor" value={factor} onChange={(e) => setFactor(e.target.value)}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px]">
            {FACTORS.map(([label, key]) => <option key={key} value={key}>{label}</option>)}
          </select>

          <button onClick={generate} disabled={generating || idx == null}
            className="mt-5 w-full min-h-11 rounded-full bg-primary py-3 font-medium text-primary-foreground transition-colors hover:bg-[color-mix(in_oklch,var(--primary),white_8%)] disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground">
            {generating ? (result ? "Updating scenarios…" : "Generating… (≈1 min)") : "Generate 30-day scenarios"}
          </button>
          {error && <p className="mt-2 text-xs text-destructive" role="alert">Generation failed. Check that the ScenarioFlow service is running, then try again.</p>}
          {inputsChanged && result && !generating && (
            <p className="mt-2 text-xs font-medium text-amber-700" role="status">
              Inputs changed. Generate again to update the displayed result.
            </p>
          )}
          <p className="sr-only" aria-live="polite">
            {generating ? (result ? "Updating scenarios" : "Generating scenarios") : result ? "Scenarios ready" : ""}
          </p>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Related historical periods shape the distribution.</p>
        </Card>

        {/* ---- output ---- */}
        <div className="relative" aria-busy={generating}>
          <motion.div
            className={`space-y-6 ${generating && result ? "pointer-events-none" : ""}`}
            animate={{ opacity: generating && result ? 0.45 : 1 }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
          >
            <div ref={chartRef} tabIndex={-1} className="scroll-mt-6 rounded-xl focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring">
              <Card className="p-6">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
                  <span className="text-sm font-medium">{FACTORS.find(([, k2]) => k2 === factor)?.[0]} · 30-day distribution{displayedDate ? ` · starts ${displayedDate}` : ""}</span>
                  <div className="flex flex-wrap items-center gap-2">
                    {result && <Badge variant="secondary">{result.support}</Badge>}
                    <button
                      type="button"
                      aria-pressed={showBase}
                      onClick={() => setShowBase((visible) => !visible)}
                      className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-background px-3 text-xs font-medium transition-colors hover:bg-secondary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                    >
                      <span className={`relative h-5 w-9 rounded-full transition-colors ${showBase ? "bg-primary" : "bg-secondary"}`} aria-hidden="true">
                        <motion.span
                          className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm"
                          animate={{ x: showBase ? 16 : 0 }}
                          transition={{ duration: shouldReduceMotion ? 0 : 0.18, ease: "easeOut" }}
                        />
                      </span>
                      Compare start-only
                    </button>
                  </div>
                </div>
                <AnimatePresence mode="wait" initial={false}>
                  {ff ? (
                    <motion.div
                      key={`${resultRevision}-${factor}`}
                      initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: shouldReduceMotion ? 0 : 0.22, ease: "easeOut" }}
                    >
                      <ScenarioFan cond={scale(ff.cond, k)} base={scale(ff.base, k)} showBase={showBase}
                        history={context?.[factor]?.map((v) => v * k)} unit={meta(factor).u} />
                    </motion.div>
                  ) : (
                    <motion.div key="empty" className="flex h-[380px] items-center justify-center text-sm text-muted-foreground">
                      {generating ? "Generating scenarios…" : "Generate to see the distribution."}
                    </motion.div>
                  )}
                </AnimatePresence>
                <p className="mt-3 text-sm text-muted-foreground">
                  Blue shows the distribution generated with language. Turn on <em>Compare start-only</em> to overlay the distribution from the same starting date without language.
                </p>
              </Card>
            </div>

            <AnimatePresence mode="wait" initial={false}>
              {result && (
                <motion.div
                  key={resultRevision}
                  initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: shouldReduceMotion ? 0 : 0.24, ease: "easeOut" }}
                >
                  <Card className="p-6">
                    <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Related periods</div>
                    <div className="mt-3 space-y-3">
                      {relatedPeriods.map((period) => (
                        <div key={period.label} className="grid grid-cols-[5rem_minmax(0,1fr)_4rem] items-center gap-x-3 gap-y-1">
                          <span className="text-sm font-medium">{period.label}</span>
                          <span className="h-1.5 overflow-hidden rounded-full bg-secondary">
                            <motion.span
                              className="block h-full rounded-full bg-primary"
                              initial={{ width: shouldReduceMotion ? `${period.matchPercent}%` : 0 }}
                              animate={{ width: `${period.matchPercent}%` }}
                              transition={{ duration: shouldReduceMotion ? 0 : 0.3, ease: "easeOut" }}
                            />
                          </span>
                          <span className="text-right text-xs tabular-nums text-muted-foreground">match {period.matchLabel}</span>
                          <span className="col-span-2 col-start-2 text-[11px] text-muted-foreground">{period.pathShareLabel}</span>
                        </div>
                      ))}
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">The bar shows narrative similarity. Path share shows how many scenarios use that period.</p>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {generating && result && (
            <div className="pointer-events-none absolute inset-x-0 top-40 z-10 flex justify-center" role="status" aria-live="polite">
              <span className="rounded-full border border-border bg-background px-4 py-2 text-sm font-medium shadow-sm">Updating scenarios…</span>
            </div>
          )}
        </div>
      </div>

      <AnimatePresence mode="wait" initial={false}>
        {result && (
          <motion.section
            key={resultRevision}
            className={`mt-8 ${generating ? "pointer-events-none" : ""}`}
            aria-label="Generated scenario results"
            initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
            animate={{ opacity: generating ? 0.45 : 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.24, ease: "easeOut" }}
          >
            <ScenarioComparisonTable
              rows={result.comparisonRows}
              startDate={displayedDate}
              selectedMarket={factor}
              onSelectMarket={viewDistribution}
            />
          </motion.section>
        )}
      </AnimatePresence>
    </main>
  );
}
