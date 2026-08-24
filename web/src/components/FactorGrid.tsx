import { useEffect, useRef, useState } from "react";
import { motion, useInView, useReducedMotion } from "framer-motion";
import { MiniFan } from "./MiniFan";

type Factor = Parameters<typeof MiniFan>[0]["factor"];
type Panel = { scenario: string; factors: Factor[] };

export function FactorGrid({ src = "/data/panel_defensive.json" }: { src?: string }) {
  const [p, setP] = useState<Panel | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const gridRef = useRef<HTMLDivElement>(null);
  const inView = useInView(gridRef, { once: true, amount: 0.25 });
  const reduced = useReducedMotion() ?? false;

  useEffect(() => {
    const controller = new AbortController();
    fetch(src, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Chart data request failed with ${response.status}`);
        return response.json() as Promise<Panel>;
      })
      .then((panel) => {
        setP(panel);
        setError(false);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(true);
      });
    return () => controller.abort();
  }, [attempt, src]);

  return (
    <div ref={gridRef}>
      {error ? (
        <div className="flex min-h-60 flex-col items-center justify-center gap-3 text-sm text-muted-foreground" role="status">
          <span>Charts unavailable.</span>
          <button
            type="button"
            onClick={() => {
              setP(null);
              setError(false);
              setAttempt((value) => value + 1);
            }}
            className="min-h-9 rounded-full border border-border px-3 text-xs font-medium transition-colors hover:border-primary/40 hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            Retry
          </button>
        </div>
      ) : !p ? (
        <div className="grid min-h-60 grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3" aria-label="Loading market charts" role="status">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className={reduced ? "" : "animate-pulse"}>
              <div className="flex items-center justify-between">
                <div className="h-3 w-16 rounded bg-muted" />
                <div className="h-3 w-10 rounded bg-muted" />
              </div>
              <div className="mt-3 h-20 rounded-md bg-[linear-gradient(to_bottom,var(--muted)_1px,transparent_1px)] bg-[size:100%_20px]">
                <div className="h-full w-full bg-[linear-gradient(145deg,transparent_42%,color-mix(in_oklab,var(--primary)_12%,transparent)_43%,color-mix(in_oklab,var(--primary)_12%,transparent)_46%,transparent_47%)]" />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <motion.div
          initial={reduced ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={reduced ? { duration: 0 } : { duration: 0.22 }}
          className="grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-3"
        >
          {p.factors.map((f) => <MiniFan key={f.label} factor={f} revealed={reduced || inView} reducedMotion={reduced} />)}
        </motion.div>
      )}
    </div>
  );
}
