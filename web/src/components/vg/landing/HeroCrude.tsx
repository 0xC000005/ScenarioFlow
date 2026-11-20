import { useCallback, useEffect, useRef, useState } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import ExplainerVideo from "@/components/vg/ExplainerVideo";
import { HatchDefs, hatch } from "@/components/vg/Hatch";
import { VG } from "@/lib/palette";
import { pts, ratioText, ratioWord, scaleFor, useShift, type ShiftMarket } from "./shift";

// The hero: the live demo's real query on 20 Jan 2016 and crude oil's next 30 days, drawn from the model's real
// output. The 30-day range from market state only (grey) turns into the range once the words are added (red);
// the grey edges stay behind, and the two day-30 brackets give the before and after side by side.

const W = 600, H = 300;
const GX = 4, GW = 492, GY = 8, GH = H - 16; // right 104 px hold the two day-30 brackets
const END = 3.6; // seconds

const clamp = (x: number) => Math.max(0, Math.min(1, x));
const io = (x: number) => {
  x = clamp(x);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};
const P = (t: number, a: number, b: number) => io((t - a) / (b - a));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

function Chart({ m, t }: { m: ShiftMarket; t: number }) {
  const [lo, hi] = scaleFor(m, 0.03, true);
  const Hn = m.hist.length;
  const X = (d: number) => GX + (GW * (d + Hn - 1)) / (Hn - 1 + 30);
  const Y = (v: number) => GY + GH * (1 - (v - lo) / (hi - lo));
  const grow = clamp(t / 0.8), k = P(t, 1.3, 3.1);
  const days = Math.max(1, Math.round(30 * grow));
  const q = (key: "p10" | "p50" | "p90") => m.base[key].map((v, d) => lerp(v, m.cond[key][d], k));
  const up = q("p90").slice(0, days + 1).map((v, d) => [X(d), Y(v)] as [number, number]);
  const dn = q("p10").slice(0, days + 1).map((v, d) => [X(d), Y(v)] as [number, number]).reverse();
  const band = `${pts(up)} ${pts(dn)}`;
  const baseBand = `${pts(m.base.p90.map((v, d) => [X(d), Y(v)]))} ${pts(m.base.p10.map((v, d) => [X(d), Y(v)]).reverse())}`;
  const ghost = P(t, 1.3, 1.7);
  const col = k < 0.5 ? "#B0B0B0" : VG.red;
  const b30 = (key: "p10" | "p90") => lerp(m.base[key][30], m.cond[key][30], P(t, 1.4, 3.3));
  const bracket = (x: number, y0: number, y1: number, c: string, o: number) => (
    <g stroke={c} strokeWidth={4} opacity={o}>
      <line x1={x} y1={y0} x2={x} y2={y1} />
      <line x1={x - 8} y1={y0} x2={x + 8} y2={y0} />
      <line x1={x - 8} y1={y1} x2={x + 8} y2={y1} />
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" aria-hidden>
      <HatchDefs id="hero" scale={0.75} />
      <HatchDefs id="hero-b" scale={1.3} />
      <line x1={X(0)} y1={GY - 6} x2={X(0)} y2={GY + GH + 6} stroke="#333" strokeWidth={2} />
      <polygon points={band} fill={hatch("hero", "white")} opacity={1 - k} />
      <polygon points={band} fill={hatch("hero", "red")} opacity={k} />
      {ghost > 0 && <polygon points={baseBand} fill={hatch("hero-b", "white")} opacity={0.55 * ghost} />}
      {m.base.paths.map((bp, i) => {
        const cp = m.cond.paths[i] || bp;
        return (
          <polyline
            key={i}
            fill="none"
            stroke={col}
            strokeWidth={1.1}
            opacity={0.6}
            points={pts(bp.slice(0, days + 1).map((v, d) => [X(d), Y(lerp(v, cp[d], k))]))}
          />
        );
      })}
      <polyline fill="none" stroke={col} strokeWidth={3.5} points={pts(q("p50").slice(0, days + 1).map((v, d) => [X(d), Y(v)]))} />
      {ghost > 0 &&
        (["p90", "p10"] as const).map((key) => (
          <polyline key={key} fill="none" stroke={VG.grey} strokeWidth={2.5} opacity={ghost} points={pts(m.base[key].map((v, d) => [X(d), Y(v)]))} />
        ))}
      <polyline fill="none" stroke={VG.white} strokeWidth={4} strokeLinejoin="round" points={pts(m.hist.map((v, i) => [X(i - Hn + 1), Y(v)]))} />
      {bracket(X(30) + 30, Y(m.base.p90[30]), Y(m.base.p10[30]), VG.grey, P(t, 0.7, 1.0))}
      {t > 1.3 && bracket(X(30) + 66, Y(b30("p90")), Y(b30("p10")), VG.red, P(t, 1.3, 1.6))}
    </svg>
  );
}

export default function HeroCrude() {
  const reduced = useReducedMotion() ?? false;
  const fig = useRef<HTMLElement>(null);
  const inView = useInView(fig, { once: true, amount: 0.35 });
  const { d, failed } = useShift();
  const [t, setT] = useState(reduced ? END : 0);
  const raf = useRef(0);

  const play = useCallback(() => {
    cancelAnimationFrame(raf.current);
    if (reduced) {
      raf.current = requestAnimationFrame(() => setT(END));
      return;
    }
    const t0 = performance.now();
    const step = (now: number) => {
      const x = Math.min(END, (now - t0) / 1000 - 0.2);
      setT(Math.max(0, x));
      if (x < END) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
  }, [reduced]);

  useEffect(() => {
    if (!d || !inView) return;
    play();
    return () => cancelAnimationFrame(raf.current);
  }, [d, inView, play]);

  // the story's chapter rail sends readers back here for chapter 0
  useEffect(() => {
    const on = () => play();
    window.addEventListener("sf:hero-replay", on);
    return () => window.removeEventListener("sf:hero-replay", on);
  }, [play]);

  const m = d?.markets.find((x) => x.name === "Crude oil");
  const after = m ? P(t, 1.3, 1.7) : 0;
  const reveal = m ? P(t, 2.6, 3.3) : 0; // the number appears only once the morph has landed, and only the real one
  const narrowed = d ? d.markets.filter((x) => ratioWord(x.ratio) === "narrower").length : 0;

  return (
    <figure ref={fig} className="w-full">
      <div className="border-l-[10px] border-vg-red bg-vg-black">
        <div className="px-4 pt-4 sm:px-6 sm:pt-5">
          <p className="text-[13px] font-semibold text-vg-red sm:text-[14px]">Your words</p>
          <p className="mt-1 text-[15px] italic leading-snug text-[#D0D0D0] sm:text-[16px]">
            {d ? `“${d.story}”` : " "}
          </p>
          <p className="mt-4 text-[17px] font-bold text-vg-white sm:text-[20px]">Crude oil, next 30 days from 20 Jan 2016</p>
        </div>
        <div className="relative px-2 pt-2 sm:px-4">
          {m ? <Chart m={m} t={t} /> : <div style={{ aspectRatio: `${W} / ${H}` }} />}
          {/* axis notes in HTML so they stay legible on phones (positions follow the chart's x-scale) */}
          <div aria-hidden className="relative mt-1 h-5 text-[12px] font-semibold text-vg-grey sm:text-[13px]">
            <span className="absolute" style={{ left: `${(100 * (GX + GW * 29 / 59)) / W + 1}%` }}>next 30</span>
            <span className="absolute right-0">day 30</span>
          </div>
          {failed && (
            <p className="absolute inset-0 grid place-items-center p-6 text-center text-[15px] text-vg-soft">
              The model output could not load.
            </p>
          )}
        </div>
        <ul className="flex flex-wrap gap-x-6 gap-y-1 px-4 pb-4 pt-2 text-[15px] font-semibold sm:px-6" aria-label="Legend">
          <li className="inline-flex items-center gap-2 text-vg-grey">
            <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden>
              <HatchDefs id="hero-lg" scale={0.75} />
              <rect x="1" y="1" width="28" height="14" fill={hatch("hero-lg", "white")} stroke={VG.grey} strokeWidth="2" />
            </svg>
            market state only
          </li>
          <li className="inline-flex items-center gap-2 text-vg-red" style={{ opacity: Math.max(0.35, after) }}>
            <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden>
              <rect x="1" y="1" width="28" height="14" fill={hatch("hero-lg", "red")} stroke={VG.red} strokeWidth="2" />
            </svg>
            + your words
          </li>
          <li className="inline-flex items-center gap-2 text-[14px] font-normal text-vg-soft">
            <svg width="30" height="16" viewBox="0 0 30 16" aria-hidden>
              <path d="M1 11 C8 4, 14 13, 29 5" fill="none" stroke="#B0B0B0" strokeWidth="1.2" />
            </svg>
            thin lines: single generated paths
          </li>
        </ul>
        <div
          className="flex flex-wrap items-end gap-x-6 gap-y-2 bg-vg-red px-4 py-4 text-vg-white transition-opacity sm:px-6 sm:py-5"
          style={{ opacity: m ? Math.max(0.15, after) : 0.15 }}
        >
          <p
            className="text-[clamp(3.25rem,7vw,5.5rem)] font-extrabold leading-[0.85] tracking-[-0.05em] tabular-nums"
            style={{ opacity: reveal }}
          >
            {m ? `×${m.ratio.toFixed(1)}` : "×"}
            <span className="ml-3 text-[0.42em] tracking-[-0.03em]">{m ? ratioWord(m.ratio) : ""}</span>
          </p>
          <p className="max-w-[17rem] pb-1 text-[15px] font-semibold leading-snug">
            crude&rsquo;s conditional distribution with words, compared to without
            <span className="mt-1 block text-[13px] font-normal opacity-90">10–90% range on day 30</span>
          </p>
          {d && (
            <a
              href="#explore"
              className="w-full pb-1 text-[14px] font-semibold underline decoration-2 underline-offset-4 hover:opacity-80 sm:ml-auto sm:w-auto"
            >
              One market of eight: {narrowed} of {d.markets.length} narrowed with the same words. All {d.markets.length} below ↓
            </a>
          )}
        </div>
      </div>
      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[14px] text-vg-soft">
        <span className="mr-auto max-w-[26rem] leading-snug">
          The model&rsquo;s real output for one query on 20 Jan 2016. The words change the range of outcomes, not a call on
          direction.
        </span>
        <button
          type="button"
          onClick={play}
          disabled={!m}
          className="inline-flex min-h-11 items-center gap-2 font-semibold text-vg-white transition-colors hover:text-vg-red disabled:opacity-40"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
            <path d="M2 7a5 5 0 1 0 1.6-3.7" fill="none" stroke="currentColor" strokeWidth="2" />
            <path d="M1.2 1.2v3.6h3.6" fill="none" stroke="currentColor" strokeWidth="2" />
          </svg>
          Replay
        </button>
        <ExplainerVideo className="inline-flex min-h-11 items-center gap-2 font-semibold text-vg-white underline decoration-vg-red decoration-2 underline-offset-[6px] transition-colors hover:text-vg-red" />
      </figcaption>
      <p className="sr-only">
        {m &&
          `Crude oil, 20 January 2016. From market state only, the 10–90% range on day 30 spans ${(m.base.p90[30] - m.base.p10[30]).toFixed(1)} dollars; with the words it spans ${(m.cond.p90[30] - m.cond.p10[30]).toFixed(1)} dollars, ${ratioText(m.ratio)}.`}
      </p>
    </figure>
  );
}
