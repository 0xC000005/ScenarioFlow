import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  generateNarrative,
  fetchHistoricalDates,
  fetchScenarioMovement,
  type NarrativePacket,
  type FactorRow,
} from "@/lib/backend";
import { Sparkline } from "@/components/Sparkline";
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
      });
      if (!isCurrentDemo2Request(requestId, generationRequestId.current)) return;
      setScenarioResult((current) => current?.requestId === requestId
        ? current
        : { requestId, submittedDate, rows: packet.scenario });
      setNarrativeResult({ requestId, submittedDate, packet });
      setGenerationStage("ready");
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

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">ScenarioFlow · Scenario → Narrative</p>
      </div>
      <h1 className="mt-2 text-display">Choose a scenario. Read the narrative.</h1>
      <p className="mt-2 max-w-2xl text-lg text-muted-foreground">
        Select an observed 30-day market episode and generate a plain-language summary of its market movements.
      </p>

      <div className="mt-8 grid gap-7 lg:grid-cols-[380px_1fr]">
        <Card className="h-fit p-6">
          <label htmlFor="d2-date" className="block text-sm font-medium">Historical scenario — start date</label>
          <select
            id="d2-date"
            value={date}
            onChange={(event) => setDate(event.target.value)}
            disabled={datesLoading || dateLoadError}
            className="mt-2 w-full rounded-lg border border-border bg-background p-2.5 text-[15px] disabled:cursor-not-allowed disabled:text-muted-foreground"
          >
            {!dates.length && (
              <option value="">{datesLoading ? "Loading dates…" : "Dates unavailable"}</option>
            )}
            {dates.map((item) => <option key={item} value={item}>{item} → +30d</option>)}
          </select>
          <p className="mt-1 text-xs text-muted-foreground">Type a year to search. The next 30 observed market days are used.</p>

          {dateLoadError && (
            <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/[0.03] p-3" role="alert">
              <p className="text-xs text-destructive">Historical dates could not be loaded.</p>
              <button
                type="button"
                onClick={() => void loadDates()}
                className="mt-2 text-xs font-medium text-foreground underline underline-offset-2"
              >
                Retry loading dates
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={generate}
            disabled={generating || !date || dateLoadError}
            className="mt-5 min-h-11 w-full rounded-full bg-primary py-3 font-medium text-primary-foreground transition-colors hover:bg-[color-mix(in_oklch,var(--primary),white_8%)] disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
          >
            {generating ? "Generating…" : "Generate narrative"}
          </button>
          {progressLabel && (
            <p className="mt-3 text-xs font-medium text-foreground" role="status" aria-live="polite">
              {progressLabel}
            </p>
          )}
          {generationError && (
            <p className="mt-2 text-xs text-destructive" role="alert">
              Generation failed. Check the ScenarioFlow service and try again.
            </p>
          )}
          {inputsChanged && (
            <p className="mt-2 text-xs font-medium text-amber-700" role="status">
              Inputs changed — generate again
            </p>
          )}
          {narrativeResult && (
            <button
              type="button"
              onClick={download}
              className="mt-3 w-full rounded-full border border-border py-2.5 text-sm font-medium transition-colors hover:bg-secondary"
            >
              Download summary (.md)
            </button>
          )}
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">A language model summarizes the market movements.</p>
        </Card>

        <div className="space-y-6" aria-busy={generating}>
          <Card className="p-6">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
              <span className="text-sm font-medium">
                Market movements{scenarioResult ? ` · ${scenarioResult.submittedDate} → +30d` : ""}
              </span>
              {scenarioResult && <Badge variant="secondary">{scenarioResult.rows.length} factors</Badge>}
            </div>
            <motion.div
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
                    <div className="hidden overflow-x-auto md:block">
                      <table className="w-full text-sm tabular-nums">
                        <thead className="text-muted-foreground">
                          <tr className="border-b border-border text-left">
                            <th className="py-1.5 pr-3 font-medium">Factor</th>
                            <th className="py-1.5 pr-3 text-right font-medium">Start</th>
                            <th className="py-1.5 pr-3 text-right font-medium">End</th>
                            <th className="py-1.5 pr-3 text-right font-medium">Δ</th>
                            <th className="py-1.5 pr-3 font-medium">Direction</th>
                            <th className="py-1.5 pr-3 font-medium">Magnitude</th>
                            <th className="py-1.5 font-medium">30-day movement</th>
                          </tr>
                        </thead>
                        <tbody>
                          {scenarioResult.rows.map((row) => (
                            <tr key={row.factor} className="border-b border-border/50">
                              <td className="py-1.5 pr-3 font-medium">{row.factor}</td>
                              <td className="py-1.5 pr-3 text-right">{row.start}</td>
                              <td className="py-1.5 pr-3 text-right">{row.end}</td>
                              <td className="py-1.5 pr-3 text-right">{row.delta}</td>
                              <td className="py-1.5 pr-3 text-foreground">
                                <span className="mr-1" aria-hidden="true">{directionArrow(row.direction)}</span>
                                {row.direction}
                              </td>
                              <td className="py-1.5 pr-3 text-muted-foreground">{row.magnitude}</td>
                              <td className="py-1.5"><Sparkline values={scenarioMovement?.[row.factor]} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="grid gap-3 md:hidden">
                      {scenarioResult.rows.map((row) => (
                        <div key={row.factor} className="rounded-lg border border-border p-3 text-sm tabular-nums">
                          <div className="flex items-center justify-between gap-3">
                            <span className="font-medium">{row.factor}</span>
                            <span className="text-foreground">
                              <span className="mr-1" aria-hidden="true">{directionArrow(row.direction)}</span>
                              {row.direction}
                            </span>
                          </div>
                          <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                            <div><span className="block text-muted-foreground">Start</span>{row.start}</div>
                            <div><span className="block text-muted-foreground">End</span>{row.end}</div>
                            <div><span className="block text-muted-foreground">Δ</span>{row.delta}</div>
                          </div>
                          <div className="mt-3 flex items-end justify-between gap-3 border-t border-border/60 pt-2">
                            <div className="text-xs"><span className="block text-muted-foreground">Magnitude</span>{row.magnitude}</div>
                            <Sparkline values={scenarioMovement?.[row.factor]} width={112} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="scenario-empty" className="flex h-48 items-center justify-center px-6 text-center text-sm text-muted-foreground">
                    {generationStage === "reading"
                      ? "Reading the factor-by-factor market movements now."
                      : "Choose an episode, then generate to inspect its factor-by-factor 30-day moves."}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </Card>

          <Card className="p-6">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Generated narrative{narrativeResult ? ` · ${narrativeResult.submittedDate}` : ""}
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
                    <NarrativePacketView md={narrativeResult.packet.markdown} />
                  </motion.div>
                ) : (
                  <motion.div key="narrative-empty" className="mt-3 flex h-40 items-center justify-center px-6 text-center text-sm text-muted-foreground">
                    {generationStage === "writing"
                      ? "The market movements are ready. The narrative is being written."
                      : "Generate an episode to compare the grounded summary with its contradictory control."}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </Card>
        </div>
      </div>
    </main>
  );
}
