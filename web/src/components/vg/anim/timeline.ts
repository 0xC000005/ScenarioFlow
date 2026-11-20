import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { animate, useInView, useReducedMotion, type AnimationPlaybackControls } from "framer-motion";

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** Local 0..1 progress of a phase [a, b] inside a 0..1 timeline. */
export const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
export const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * One orchestrated, linear 0..1 timeline that plays once when `ref` scrolls into view.
 * Before it plays, the figure holds a meaningful `poster` frame (real content, never an empty
 * panel); the first play continues from that frame instead of blanking it. It starts once the
 * figure's top has crossed 65% of the viewport height (i.e. it fills ~35% of the screen), which —
 * unlike an element-fraction threshold — also fires for figures taller than the viewport.
 * Reduced motion: pinned to the final frame. `replay()` restarts from 0; `run(from, dur)` plays a
 * short tail (used when the viewer switches the market or view).
 */
export function useTimeline(ref: RefObject<Element | null>, duration: number, ready: boolean, poster = 0) {
  const reduced = useReducedMotion() ?? false;
  const inView = useInView(ref, { once: true, amount: "some", margin: "0px 0px -35% 0px" });
  const [t, setT] = useState(poster);
  const controls = useRef<AnimationPlaybackControls | null>(null);
  const started = useRef(false);

  const run = useCallback(
    (from: number, dur: number) => {
      controls.current?.stop();
      if (reduced) return; // the returned t is pinned to 1
      controls.current = animate(from, 1, { duration: dur, ease: "linear", onUpdate: setT });
    },
    [reduced],
  );

  useEffect(() => {
    if (!ready || started.current || reduced) return;
    if (inView) {
      started.current = true;
      run(poster, duration * (1 - poster));
    }
  }, [ready, inView, reduced, run, duration, poster]);

  useEffect(() => () => controls.current?.stop(), []);

  const replay = useCallback(() => run(0, duration), [run, duration]);
  return { t: reduced ? 1 : t, run, replay, reduced };
}

/** Container width via ResizeObserver (SVGs render at true pixel width for crisp type). */
export function useWidth<T extends Element>(fallback = 900): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cw = entries[0]?.contentRect.width;
      if (cw) setW(Math.round(cw));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Fetch a JSON file from /public, reporting failure (non-ok response, network or parse error) and offering a retry. */
export function useJsonResource<T>(url: string): { data: T | null; failed: boolean; retry: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<T>;
      })
      .then((d) => {
        if (!alive) return;
        if (d) setData(d);
        else setFailed(true);
      })
      .catch((e) => {
        if (!alive) return;
        console.error(`[figure] could not load ${url}:`, e);
        setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [url, attempt]);
  const retry = useCallback(() => {
    setFailed(false);
    setAttempt((a) => a + 1);
  }, []);
  return { data, failed, retry };
}

/** Fetch a JSON file from /public once (null until loaded). */
export function useJson<T>(url: string): T | null {
  return useJsonResource<T>(url).data;
}
