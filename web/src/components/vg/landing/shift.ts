import { useEffect, useState } from "react";

// The live demo's 20 Jan 2016 query: the same file the explainer video and the scroll story draw from
// (public/story/data_v2.js holds identical numbers). base = market state only, cond = market state + the words.

export type Band = { p10: number[]; p50: number[]; p90: number[]; paths: number[][] };
export type ShiftMarket = { key: string; name: string; hist: number[]; base: Band; cond: Band; ratio: number };
export type Shift = {
  source: string;
  story: string;
  as_of: string;
  precedents: { date: string; cosine: number }[];
  markets: ShiftMarket[];
};

export const SHIFT_SRC = "/data/narrative_shift_2016-01-20.json";

let cache: Promise<Shift> | null = null; // one fetch per page load for the hero and the grid (static file, not a result cache)

export function useShift() {
  const [d, setD] = useState<Shift | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    if (!cache) {
      cache = fetch(SHIFT_SRC).then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json() as Promise<Shift>;
      });
      cache.catch(() => {
        cache = null;
      });
    }
    cache.then((x) => alive && setD(x)).catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, []);
  return { d, failed };
}

/** y-range over everything drawn for one market (the explainer's scaleFor), padded by 3%. */
export function scaleFor(m: ShiftMarket, padK = 0.03, paths = false): [number, number] {
  const all = [...m.hist, ...m.base.p10, ...m.base.p90, ...m.cond.p10, ...m.cond.p90];
  if (paths) all.push(...m.base.paths.flat(), ...m.cond.paths.flat());
  const lo = Math.min(...all), hi = Math.max(...all), p = (hi - lo) * padK;
  return [lo - p, hi + p];
}

/**
 * "×1.7 wider" / "×0.8 narrower": the day-30 10–90% range with the words over the range from market state only.
 * The word is decided on the number as shown, so a ratio that rounds to 1 reads "unchanged", never "×1.0 narrower".
 */
export const ratioWord = (r: number, digits = 1) => {
  const shown = Number(r.toFixed(digits));
  return shown === 1 ? "unchanged" : shown > 1 ? "wider" : "narrower";
};
export const ratioText = (r: number, digits = 1) => `×${r.toFixed(digits)} ${ratioWord(r, digits)}`;

export const pts = (a: number[][]) => a.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");

export const fmtDate = (iso: string) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
