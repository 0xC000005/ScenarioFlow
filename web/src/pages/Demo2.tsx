import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  generateNarrative,
  fetchHistoricalDates,
  fetchScenarioMovement,
  type NarrativePacket,
  type FactorRow,
} from "@/lib/backend";
import { Sparkline } from "@/components/Sparkline";
import { DemoHeader } from "@/components/DemoHeader";
import { ControlStep, GenerateButton } from "@/components/DemoControls";
import { HatchDefs, hatch } from "@/components/vg/Hatch";
import { NarrativePacketView } from "@/components/NarrativePacket";
import {
  captureCoherentDemo2Snapshot,
  demo2DownloadFilename,
  demo2InputsChanged,
  directionArrow,
  isCurrentDemo2Request,
  restoreDemo2FailureResults,
} from "@/demo2Presentation";

type ScenarioResult = {
  requestId: number;
  submittedDate: string;
  rows: FactorRow[];
};

type NarrativeResult = {
  requestId: number;
  submittedDate: string;
  packet: NarrativePacket;
};

type ActiveRequest = { requestId: number; submittedDate: string };
type GenerationStage = "idle" | "reading" | "writing" | "ready" | "failed";

const DEFAULT_DATE = "2008-09-22";

export default function Demo2() {
  const shouldReduceMotion = useReducedMotion() ?? false;
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [datesLoading, setDatesLoading] = useState(true);
  const [dateLoadError, setDateLoadError] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStage, setGenerationStage] = useState<GenerationStage>("idle");
  const [generationError, setGenerationError] = useState(false);
  const [activeRequest, setActiveRequest] = useState<ActiveRequest | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);
  const [movementByRequest, setMovementByRequest] = useState<
    Record<number, Record<string, number[]>>
  >({});
  const [narrativeResult, setNarrativeResult] = useState<NarrativeResult | null>(null);
  // wall-clock marks of the live run: [start, scenario table published, narrative returned]
  const [marks, setMarks] = useState<{ start: number | null; read: number | null; done: number | null }>({ start: null, read: null, done: null });
  const now = useClock(generating);
  const generationRequestId = useRef(0);
  const dateLoadRequestId = useRef(0);

  const loadDates = useCallback(async () => {
    const requestId = ++dateLoadRequestId.current;
    setDatesLoading(true);
    setDateLoadError(false);
    try {
      const nextDates = await fetchHistoricalDates();
      if (requestId !== dateLoadRequestId.current) return;
      if (!nextDates.length) throw new Error("No historical dates returned");
      setDates(nextDates);
      setDate((current) => current && nextDates.includes(current)
        ? current
        : (nextDates.find((item) => item >= DEFAULT_DATE)
          ?? nextDates[Math.floor(nextDates.length / 2)]));
    } catch (error) {
      if (requestId !== dateLoadRequestId.current) return;
      console.error("Unable to load historical dates:", error);
      setDateLoadError(true);
    } finally {
      if (requestId === dateLoadRequestId.current) setDatesLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void loadDates(), 0);
    return () => {
      window.clearTimeout(initialLoad);
      dateLoadRequestId.current += 1;
      generationRequestId.current += 1;
    };
  }, [loadDates]);

  async function generate() {
    if (!date) return;
    const submittedDate = date;
    const requestId = ++generationRequestId.current;
    const previousSnapshot = captureCoherentDemo2Snapshot(
      scenarioResult,
      narrativeResult,
    );
    let publishedScenario: ScenarioResult | null = null;
    setActiveRequest({ requestId, submittedDate });
    setGenerating(true);
    setGenerationStage("reading");
    setGenerationError(false);
    setMarks({ start: Date.now(), read: null, done: null });

    void fetchScenarioMovement(submittedDate)
      .then((movement) => {
        if (!isCurrentDemo2Request(requestId, generationRequestId.current)) return;
        setMovementByRequest((current) => ({ ...current, [requestId]: movement }));
      })
      .catch((error) => {
        if (isCurrentDemo2Request(requestId, generationRequestId.current)) {
          console.error("Unable to load scenario movement:", error);
        }
      });

    try {
      const packet = await generateNarrative(submittedDate, (rows) => {
        if (!isCurrentDemo2Request(requestId, generationRequestId.current)) return;
        publishedScenario = { requestId, submittedDate, rows };
        setScenarioResult(publishedScenario);
        setGenerationStage("writing");
        setMarks((m) => ({ ...m, read: Date.now() }));
      });
      if (!isCurrentDemo2Request(requestId, generationRequestId.current)) return;
      setScenarioResult((current) => current?.requestId === requestId
        ? current
        : { requestId, submittedDate, rows: packet.scenario });
      setNarrativeResult({ requestId, submittedDate, packet });
      setGenerationStage("ready");
      setMarks((m) => ({ ...m, read: m.read ?? Date.now(), done: Date.now() }));
    } catch (error) {
      if (!isCurrentDemo2Request(requestId, generationRequestId.current)) return;
      console.error("[live] generateNarrative failed:", error);
      const restored = restoreDemo2FailureResults(
        previousSnapshot,
        publishedScenario,
      );
      setScenarioResult(restored.scenario);
      setNarrativeResult(restored.narrative);
      setGenerationError(true);
      setGenerationStage("failed");
    } finally {
      if (isCurrentDemo2Request(requestId, generationRequestId.current)) {
        setGenerating(false);
        setActiveRequest(null);
      }
    }
  }

  function download() {
    if (!narrativeResult) return;
    const blob = new Blob([narrativeResult.packet.markdown], { type: "text/markdown" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = demo2DownloadFilename(narrativeResult.submittedDate);
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  const scenarioMovement = scenarioResult
    ? movementByRequest[scenarioResult.requestId]
    : undefined;
  const scenarioIsPrevious = Boolean(
    generating && activeRequest && scenarioResult
    && activeRequest.requestId !== scenarioResult.requestId,
  );
  const narrativeIsPrevious = Boolean(
    generating && activeRequest && narrativeResult
    && activeRequest.requestId !== narrativeResult.requestId,
  );
  const visibleResultDates = [
    scenarioResult?.submittedDate ?? null,
    narrativeResult?.submittedDate ?? null,
  ];
  const inputsChanged = demo2InputsChanged(
    date,
    activeRequest?.submittedDate ?? null,
    visibleResultDates,
  );
  const progressLabel = generationStage === "reading"
    ? "1 of 2 · Reading market movements…"
    : generationStage === "writing"
      ? "2 of 2 · Writing narrative…"
      : generationStage === "ready"
        ? "Narrative ready"
        : "";

  const largeMovers = scenarioResult
    ? scenarioResult.rows.filter((row) => isLargeMove(row)).length
    : 0;
  const readingState: StepState = generationStage === "reading"
    ? "active"
    : generationStage === "writing" || generationStage === "ready" ? "done" : "pending";
  const writingState: StepState = generationStage === "writing"
    ? "active"
    : generationStage === "ready" ? "done" : "pending";
  const secs = (a: number | null, b: number | null) => (a == null ? null : Math.max(0, ((b ?? now) - a) / 1000));
  const readingSecs = readingState === "pending" ? null : secs(marks.start, marks.read);
  const writingSecs = writingState === "pending" ? null : secs(marks.read, marks.done);
  const totalSecs = secs(marks.start, generating ? null : marks.done);
  const largeNames = scenarioResult ? scenarioResult.rows.filter((row) => isLargeMove(row)).map((row) => row.factor) : [];

  return (
    <main className="bg-vg-white text-vg-black">
      <DemoHeader
        kicker="Scenario → Narrative"
        line1="Choose a scenario."
        line2="Read the narrative."
        lede="Select an observed 30-day market episode and generate a plain-language summary of its market movements."
        steps={[
          { title: "Episode", detail: "Pick the start date of an observed 30-day market episode." },
          { title: "Market movements", detail: "Read its factor-by-factor moves." },
          { title: "Generated narrative", detail: "A language model writes the narrative and a contrast narrative written for a different period." },
        ]}
      />

      <div className="mx-auto grid max-w-7xl gap-12 px-4 py-12 sm:px-8 sm:py-16 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-14">
        {/* Control panel */}
        <aside className="h-fit lg:sticky lg:top-20">
          <ControlStep n={1} label="Episode start date" htmlFor="d2-date" first>
            <div className="relative mt-3">
              <select
                id="d2-date"
                value={date}
                onChange={(event) => setDate(event.target.value)}
                disabled={datesLoading || dateLoadError}
                className="h-12 w-full appearance-none rounded-none border-2 border-vg-black bg-vg-white pl-3 pr-10 text-[15px] font-medium tabular-nums text-vg-black outline-none focus-visible:border-vg-red disabled:cursor-not-allowed disabled:border-vg-soft disabled:text-vg-grey"
              >
                {!dates.length && (
                  <option value="">{datesLoading ? "Loading dates…" : "Dates unavailable"}</option>
                )}
                {dates.map((item) => <option key={item} value={item}>{item} → +30d</option>)}
              </select>
              <svg aria-hidden="true" width="12" height="8" viewBox="0 0 12 8" className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
                <path d="M1 1l5 5 5-5" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="square" />
              </svg>
            </div>
            <p className="mt-2 text-[13px] leading-snug text-vg-mute">Type a year to search. The next 30 observed market days are used.</p>

            {dateLoadError && (
              <div className="mt-4 border-l-[6px] border-vg-red bg-vg-paper p-3" role="alert">
                <p className="text-[13px] font-semibold text-vg-red">Historical dates could not be loaded.</p>
                <button
                  type="button"
                  onClick={() => void loadDates()}
                  className="mt-2 text-[13px] font-bold text-vg-black underline decoration-2 underline-offset-4 hover:text-vg-red"
                >
                  Retry loading dates
                </button>
              </div>
            )}
          </ControlStep>

          <GenerateButton
            onClick={generate}
            disabled={generating || !date || dateLoadError}
            busy={generating}
            label="Generate narrative"
            busyLabel="Generating…"
            elapsed={totalSecs ?? undefined}
          />

          <ol className="mt-5 border-t-2 border-vg-black" aria-label="Generation steps">
            <StageRow index={1} label="Reading market movements" state={readingState} reduced={shouldReduceMotion} secs={readingSecs} />
            <StageRow index={2} label="Writing narrative" state={writingState} reduced={shouldReduceMotion} secs={writingSecs} />
          </ol>
          {progressLabel && (
            <p className="mt-3 flex items-baseline justify-between gap-3 text-[13px] font-semibold text-vg-black">
              <span role="status" aria-live="polite">{progressLabel}</span>
              {totalSecs != null && <span className="tabular-nums text-vg-mute">{Math.floor(totalSecs)} s total</span>}
            </p>
          )}
          {generationError && (
            <p className="mt-3 border-l-[6px] border-vg-red pl-3 text-[13px] leading-snug text-vg-red" role="alert">
              Generation failed. Check the ScenarioFlow service and try again; any existing result remains available.
            </p>
          )}
          {inputsChanged && (
            <p className="mt-3 border-l-[6px] border-vg-black pl-3 text-[13px] font-bold text-vg-black" role="status">
              Inputs changed — generate again
            </p>
          )}
          {narrativeResult && (
            <button
              type="button"
              onClick={download}
              className="mt-4 min-h-11 w-full border-2 border-vg-black text-[14px] font-bold transition-colors hover:bg-vg-black hover:text-vg-white"
            >
              Download summary (.md)
            </button>
          )}
          <p className="mt-4 text-[13px] leading-relaxed text-vg-mute">A language model summarizes the market movements.</p>
        </aside>

        <div className="min-w-0" aria-busy={generating}>
          {/* Market movements */}
          <section className="border-t-8 border-vg-black pt-5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
              <h2 className="flex items-baseline gap-3 text-[30px] leading-tight font-extrabold tracking-[-0.03em] sm:text-[36px]">
                <span aria-hidden="true" className="text-vg-red">2</span>Market movements
              </h2>
              {scenarioResult && (
                <span className="text-[15px] font-semibold text-vg-mute tabular-nums">
                  {scenarioResult.submittedDate} → +30d · {scenarioResult.rows.length} factors
                </span>
              )}
            </div>
            <motion.div
              className="mt-6"
              animate={{ opacity: scenarioIsPrevious ? 0.5 : 1 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            >
              <AnimatePresence mode="wait" initial={false}>
                {scenarioResult ? (
                  <motion.div
                    key={scenarioResult.requestId}
                    initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: shouldReduceMotion ? 0 : 0.23, ease: "easeOut" }}
                  >
                    <div className="grid sm:grid-cols-[auto_minmax(0,1fr)]">
                      <div className="flex items-end gap-4 bg-vg-red px-5 py-5 text-vg-white sm:px-7">
                        <span className="text-vg-mega tabular-nums">
                          {largeMovers}
                          <span className="text-[0.4em] tracking-[-0.02em] text-vg-white/75">/{scenarioResult.rows.length}</span>
                        </span>
                      </div>
                      <div className="flex flex-col justify-end gap-2 bg-vg-black px-5 py-5 text-vg-white sm:px-7">
                        <p className="text-[18px] font-extrabold leading-snug tracking-[-0.01em] sm:text-[20px]">
                          factors made a large 30-day move in this episode
                        </p>
                        <p className="text-[14px] leading-snug text-vg-soft">
                          {largeNames.length ? `Large: ${largeNames.join(", ")}. ` : ""}Magnitude labels come from the service; red marks the large moves, ▲/▼ the direction.
                        </p>
                      </div>
                    </div>

                    <div className="hidden md:block">
                      <table className="w-full text-[14px] tabular-nums">
                        <thead>
                          <tr className="border-b-2 border-vg-black text-left text-[13px] text-vg-mute">
                            <th className="py-2.5 pr-3 pl-4 font-semibold">Factor</th>
                            <th className="py-2.5 pr-3 text-right font-semibold">Start</th>
                            <th className="py-2.5 pr-3 text-right font-semibold">End</th>
                            <th className="py-2.5 pr-3 text-right font-semibold">Δ</th>
                            <th className="py-2.5 pr-3 font-semibold">Direction</th>
                            <th className="py-2.5 pr-3 font-semibold">Magnitude</th>
                            <th className="py-2.5 font-semibold">30-day movement</th>
                          </tr>
                        </thead>
                        <tbody>
                          {scenarioResult.rows.map((row, i) => {
                            const large = isLargeMove(row);
                            return (
                              <tr key={row.factor} className="border-b border-vg-rule">
                                <td className="relative py-2.5 pr-3 pl-4 font-bold">
                                  {large && <span aria-hidden className="absolute top-1.5 bottom-1.5 left-0 w-1.5 bg-vg-red" />}
                                  {row.factor}
                                </td>
                                <td className="py-2.5 pr-3 text-right text-vg-mute">{row.start}</td>
                                <td className="py-2.5 pr-3 text-right">{row.end}</td>
                                <td className={"py-2.5 pr-3 text-right font-bold " + (large ? "text-vg-red" : "")}>{row.delta}</td>
                                <td className="py-2.5 pr-3">
                                  <span className="mr-1.5 text-[11px]" aria-hidden="true">{glyph(row.direction)}</span>
                                  {row.direction}
                                </td>
                                <td className={"py-2.5 pr-3 " + magnitudeClass(row)}>{row.magnitude}</td>
                                <td className="py-1.5">
                                  <Sparkline
                                    values={scenarioMovement?.[row.factor]}
                                    width={150}
                                    height={30}
                                    tone={sparkTone(row)}
                                    drawDelay={shouldReduceMotion ? undefined : i * 0.04}
                                  />
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <div className="grid md:hidden">
                      {scenarioResult.rows.map((row, i) => {
                        const large = isLargeMove(row);
                        return (
                          <div key={row.factor} className={"border-b border-vg-rule py-4 text-[14px] tabular-nums " + (large ? "border-l-[6px] border-l-vg-red pl-3" : "pl-0")}>
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-bold">{row.factor}</span>
                              <span>
                                <span className="mr-1.5 text-[11px]" aria-hidden="true">{glyph(row.direction)}</span>
                                {row.direction}
                              </span>
                            </div>
                            <div className="mt-2 grid grid-cols-3 gap-2 text-[13px]">
                              <div><span className="block text-vg-mute">Start</span>{row.start}</div>
                              <div><span className="block text-vg-mute">End</span>{row.end}</div>
                              <div><span className="block text-vg-mute">Δ</span><span className={large ? "font-bold text-vg-red" : "font-bold"}>{row.delta}</span></div>
                            </div>
                            <div className="mt-3 flex items-end justify-between gap-3">
                              <div className="text-[13px]"><span className="block text-vg-mute">Magnitude</span><span className={magnitudeClass(row)}>{row.magnitude}</span></div>
                              <Sparkline
                                values={scenarioMovement?.[row.factor]}
                                width={128}
                                height={30}
                                tone={sparkTone(row)}
                                drawDelay={shouldReduceMotion ? undefined : i * 0.04}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </motion.div>
                ) : (
                  generationStage === "reading" ? (
                    <motion.div key="scenario-reading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <WaitPanel
                        secs={readingSecs ?? 0}
                        step="Step 1 of 2"
                        title="Reading market movements"
                        detail={`The service loads the observed 30 market days from ${activeRequest?.submittedDate ?? date} and summarizes each factor's start, end, change, direction and magnitude.`}
                        reduced={shouldReduceMotion}
                      />
                    </motion.div>
                  ) : (
                    <motion.div key="scenario-empty" className="grid min-h-56 border-[3px] border-vg-black sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                      <div className="flex flex-col justify-center gap-4 p-6 sm:p-10">
                        <p className="text-[32px] font-extrabold leading-tight tracking-[-0.03em]">No episode yet.</p>
                        <p className="max-w-md text-[15px] text-vg-mute">Choose an episode start date, then generate to inspect its factor-by-factor 30-day moves and the narrative written from them.</p>
                      </div>
                      <EmptyHatch />
                    </motion.div>
                  )
                )}
              </AnimatePresence>
            </motion.div>
          </section>

          {/* Narrative */}
          <section className="mt-16 border-t-8 border-vg-black pt-5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
              <h2 className="flex items-baseline gap-3 text-[30px] leading-tight font-extrabold tracking-[-0.03em] sm:text-[36px]">
                <span aria-hidden="true" className="text-vg-red">3</span>Generated narrative
              </h2>
              {narrativeResult && (
                <span className="text-[15px] font-semibold text-vg-mute tabular-nums">{narrativeResult.submittedDate}</span>
              )}
            </div>
            <motion.div
              animate={{ opacity: narrativeIsPrevious ? 0.5 : 1 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            >
              <AnimatePresence mode="wait" initial={false}>
                {narrativeResult ? (
                  <motion.div
                    key={narrativeResult.requestId}
                    initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: shouldReduceMotion ? 0 : 0.23, ease: "easeOut" }}
                  >
                    <div data-testid="d2-narrative"><NarrativePacketView md={narrativeResult.packet.markdown} /></div>
                  </motion.div>
                ) : (
                  generationStage === "writing" ? (
                    <motion.div key="narrative-writing" className="mt-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <WaitPanel
                        secs={writingSecs ?? 0}
                        step="Step 2 of 2"
                        title="Writing the narrative"
                        detail="A language model is writing several narrative formats from the factor moves above, each paired with a contrast narrative written for a different period. This live step usually takes one to three minutes."
                        reduced={shouldReduceMotion}
                      />
                    </motion.div>
                  ) : (
                    <motion.div key="narrative-empty" className="mt-6 grid min-h-40 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
                      <div className="relative flex items-center bg-vg-black py-8 pl-9 pr-6 text-[15px] text-vg-soft">
                        <span aria-hidden className="absolute inset-y-0 left-0 w-2.5 bg-vg-red" />
                        {generationStage === "reading"
                          ? "The narrative is written once the market movements are read."
                          : "Generate an episode to compare the grounded summary with a contrast narrative written for a different period."}
                      </div>
                      <div className="flex items-center border-2 border-vg-rule bg-vg-paper px-6 py-8 text-[14px] text-vg-mute lg:border-l-0">
                        Contrast narrative: written for a different period
                      </div>
                    </motion.div>
                  )
                )}
              </AnimatePresence>
            </motion.div>
          </section>
        </div>
      </div>
    </main>
  );
}

type StepState = "pending" | "active" | "done";

// ▲/▼ carry the direction (mapped from directionArrow's ↑/↓/→); colour is never used for "down".
function glyph(direction: string) {
  const arrow = directionArrow(direction);
  if (arrow === "↑") return "▲";
  if (arrow === "↓") return "▼";
  // spread labels: wider = the level rose, tighter/narrower = it fell
  const d = direction.trim().toLowerCase();
  if (d === "wider" || d === "widening") return "▲";
  if (d === "tighter" || d === "narrower" || d === "tightening") return "▼";
  return "■";
}

// Current time while `running`, ticking 4× a second (drives the elapsed-seconds readouts).
function useClock(running: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [running]);
  return now;
}

function WaitPanel({ secs, step, title, detail, reduced }: { secs: number; step: string; title: string; detail: string; reduced: boolean }) {
  return (
    <div className="grid min-h-56 bg-vg-black text-vg-white sm:grid-cols-[minmax(0,15rem)_minmax(0,1fr)]">
      <div className="flex flex-col justify-between gap-4 border-b-[3px] border-vg-red p-6 sm:border-b-0 sm:border-r-[3px]">
        <p className="text-[13px] font-bold uppercase tracking-[0.08em] text-vg-soft">{step} · elapsed</p>
        <p className="text-vg-mega tabular-nums" aria-hidden="true">
          {Math.floor(secs)}<span className="text-[0.4em] tracking-[-0.02em] text-vg-red"> s</span>
        </p>
      </div>
      <div className="flex flex-col justify-center gap-4 p-6">
        <p className="text-[24px] font-extrabold leading-tight tracking-[-0.03em]">{title}</p>
        <p className="max-w-xl text-[14px] leading-relaxed text-vg-soft">{detail}</p>
        <div aria-hidden="true" className="relative h-3 w-full max-w-xl overflow-hidden bg-vg-panel">
          <motion.div
            className="absolute inset-y-0 -left-6 right-0"
            style={{ backgroundImage: "repeating-linear-gradient(45deg, var(--vg-red) 0 3px, transparent 3px 9px)" }}
            animate={reduced ? undefined : { x: [0, 12.7] }}
            transition={reduced ? undefined : { duration: 0.6, ease: "linear", repeat: Infinity }}
          />
        </div>
      </div>
    </div>
  );
}

// Empty-state panel: the same parts as Demo 1's (grey 45° hatch = observed market state), decoration only.
function EmptyHatch() {
  return (
    <div aria-hidden="true" className="relative hidden border-l-[3px] border-vg-black sm:block">
      <svg className="absolute inset-0 h-full w-full">
        <HatchDefs id="d2-empty" />
        <rect x="0" y="0" width="100%" height="100%" fill={hatch("d2-empty", "grey")} />
      </svg>
      <span className="absolute bottom-3 left-3 bg-vg-white px-2 py-1 text-[12px] font-bold text-vg-mute">Observed market</span>
    </div>
  );
}

// Large/flat come from the backend's own magnitude label (flat | small | medium | large).
function isLargeMove(row: FactorRow) {
  return row.magnitude.trim().toLowerCase() === "large";
}

function sparkTone(row: FactorRow): "red" | "grey" | "black" {
  if (isLargeMove(row)) return "red";
  return row.magnitude.trim().toLowerCase() === "flat" ? "grey" : "black";
}

function magnitudeClass(row: FactorRow) {
  if (isLargeMove(row)) return "font-bold text-vg-red";
  return row.magnitude.trim().toLowerCase() === "flat" ? "text-vg-grey" : "text-vg-black";
}

function StageRow({ index, label, state, reduced, secs }: { index: number; label: string; state: StepState; reduced: boolean; secs: number | null }) {
  const tone = state === "active"
    ? "bg-vg-red text-vg-white"
    : state === "done" ? "bg-vg-black text-vg-white" : "bg-vg-white text-vg-grey";
  return (
    <li className={"relative flex min-h-11 items-center gap-3 overflow-hidden border-b-2 border-vg-black px-3 text-[14px] font-bold transition-colors " + tone}>
      {state === "active" && !reduced && (
        <motion.span
          aria-hidden
          className="absolute inset-y-0 left-0 w-1/3 bg-vg-red-deep"
          initial={{ x: "-100%" }}
          animate={{ x: "300%" }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
        />
      )}
      <span className="relative w-4 tabular-nums">{index}</span>
      <span className="relative flex-1">{label}</span>
      <span className="relative text-[13px] font-semibold tabular-nums">
        {state === "done" ? "Done" : state === "active" ? "Running" : ""}
        {state !== "pending" && secs != null ? ` · ${Math.floor(secs)} s` : ""}
      </span>
    </li>
  );
}
