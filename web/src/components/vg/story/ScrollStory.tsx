import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useReducedMotion } from "framer-motion";
import { loadStory, prefetchStory, type SFStoryApi } from "./engine";
import { makeDrawer } from "./stage";
import { CHAPTERS, CLOSE_LINE, STORY_END, STORY_START, buildKnots, chapterIndex, tToU, uToT, type Beat } from "./chapters";
import "./story.css";

// The explainer video, told as you scroll. Every frame is drawn live by the video's own render code
// (public/story/engine.js) from the same real data; scroll position is mapped to the video's clock.

const { knots, total } = buildKnots();

type Line = { t0: number; t1?: number; text: string; tone: Beat["tone"]; ch: number; echo?: boolean };

const pad2 = (n: number) => String(n).padStart(2, "0");
const clock = (s: number) => `${Math.floor(s / 60)}:${pad2(Math.floor(s % 60))}`;

function resolve(text: string) {
  const crude = window.DATA?.markets?.[6]?.ratio;
  const v16 = (window as unknown as { V16?: { neigh: { hot: number[]; share_all: number } } }).V16;
  return text
    .replace("{CRUDE}", crude ? crude.toFixed(1) : "1.7")
    .replace("{HIT}", v16 ? String(v16.neigh.hot.reduce((a, b) => a + b, 0)) : "6")
    .replace("{SHARE}", v16 ? String(Math.round(100 * v16.neigh.share_all)) : "15");
}

/** Captions recorded from the video's own caption strip, plus the frame-text beats for scenes without one. */
function buildLines(api: SFStoryApi): Line[] {
  const lines: Line[] = api.captions.map((c) => ({ t0: c.t0, t1: c.t1, text: c.text, tone: c.tone, ch: chapterIndex(c.t0) }));
  CHAPTERS.forEach((c, i) => c.beats.forEach((b) => lines.push({ t0: b.t0, text: resolve(b.text), tone: b.tone, ch: i, echo: b.echo })));
  return lines.sort((a, b) => a.t0 - b.t0);
}

/**
 * The line to show at t: the latest one that has started in the current chapter (held until the next).
 * On wide screens, beats that echo legible frame text are skipped so each screen has one voice.
 */
function lineAt(lines: Line[], t: number, wide: boolean) {
  const ch = chapterIndex(t);
  let idx = -1;
  lines.forEach((l, i) => {
    if (l.ch === ch && l.t0 <= t + 1e-6 && !(wide && l.echo)) idx = i;
  });
  return idx;
}

/** Story time a chapter jump lands on: its first visible line, else the frame shown under reduced motion. */
function jumpTime(lines: Line[], i: number, wide: boolean) {
  const first = lines.find((l) => l.ch === i && !(wide && l.echo));
  const start = Math.max(CHAPTERS[i].t, STORY_START);
  return Math.max(start + 0.02, Math.min(first ? first.t0 + 0.05 : CHAPTERS[i].still, CHAPTERS[i].still));
}

const WIDE_Q = "(min-width: 1024px)";
function useWide() {
  const [wide, setWide] = useState(() => typeof window !== "undefined" && window.matchMedia(WIDE_Q).matches);
  useEffect(() => {
    const mq = window.matchMedia(WIDE_Q);
    const on = () => setWide(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return wide;
}

/** Scroll distance over which the stage stays pinned (the sticky box sits below the fixed nav). */
function pinRange(sec: HTMLElement, st: HTMLElement) {
  const top = parseFloat(getComputedStyle(st).top) || 0;
  return Math.max(1, sec.offsetHeight - st.offsetHeight - top);
}

const toneCls = (tone: Beat["tone"], dark: boolean) =>
  tone === "red" ? "text-vg-red" : tone === "mute" ? (dark ? "text-vg-soft" : "text-vg-mute") : dark ? "text-vg-white" : "text-vg-black";

/**
 * True once the story's ≈2.4 MB of data should run: when the section is about to enter the viewport. A few
 * seconds after load the files are only prefetched (downloaded, not executed), so a reader who stays in the
 * hero never pays the parse cost on the main thread.
 */
function useNear(ref: React.RefObject<Element | null>) {
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    const now = new IntersectionObserver((e) => e.some((x) => x.isIntersecting) && setNear(true), { rootMargin: "0px 0px 25% 0px" });
    const timer = window.setTimeout(prefetchStory, 4000);
    now.observe(el);
    return () => {
      now.disconnect();
      window.clearTimeout(timer);
    };
  }, [ref, near]);
  return near;
}

function useStoryApi(enabled: boolean) {
  const [api, setApi] = useState<SFStoryApi | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    loadStory()
      .then((a) => alive && setApi(a))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [enabled]);
  return { api, failed };
}

function Stage({ gRef, defs, label, className = "" }: { gRef: React.RefObject<SVGGElement | null>; defs?: string; label: string; className?: string }) {
  return (
    <svg className={"sf-stage block aspect-[4/5] bg-vg-black " + className} viewBox="0 0 1080 1350" role="img" aria-label={label}>
      {defs ? <defs dangerouslySetInnerHTML={{ __html: defs }} /> : null}
      <g ref={gRef} />
    </svg>
  );
}

function CloseLinks({ dark }: { dark: boolean }) {
  const base = "inline-flex min-h-11 items-center justify-center whitespace-nowrap px-2 text-[14px] tracking-[-0.01em] font-semibold transition-colors sm:justify-start sm:px-4 sm:text-[15px] ";
  return (
    <div className="mt-3 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap lg:mt-6">
      <Link to="/demo1" className={base + "bg-vg-red text-vg-white hover:bg-vg-red-deep"}>Narrative → Scenario</Link>
      <Link to="/demo2" className={base + (dark ? "border-2 border-vg-white text-vg-white hover:bg-vg-white hover:text-vg-black" : "border-2 border-vg-black hover:bg-vg-black hover:text-vg-white")}>
        Scenario → Narrative
      </Link>
      <a
        href="#paper"
        className={
          "col-span-2 inline-flex min-h-11 items-center text-[14px] font-semibold underline decoration-vg-red decoration-2 underline-offset-[6px] sm:px-4 sm:text-[15px] " +
          (dark ? "text-vg-white" : "")
        }
      >
        Poster and paper ↓
      </a>
    </div>
  );
}

const TRANSCRIPT_LABEL = "The explainer as text";

function Transcript({ lines }: { lines: Line[] }) {
  return (
    <div className="sr-only">
      <h3>{TRANSCRIPT_LABEL}</h3>
      <ol>
        {CHAPTERS.slice(1).map((c, k) => (
          <li key={c.t}>
            {c.title}:{" "}
            {lines.filter((l) => l.ch === k + 1).map((l) => l.text).join(" ")}
          </li>
        ))}
      </ol>
    </div>
  );
}

/* ───────────────────────── reduced motion: one still per chapter, with its lines ───────────────────────── */

function Still({ api, t, idx, title }: { api: SFStoryApi; t: number; idx: number; title: string }) {
  const gRef = useRef<SVGGElement>(null);
  const pfx = `st${idx}-`;
  useEffect(() => {
    if (gRef.current) makeDrawer(gRef.current, api, { captions: false, idPrefix: pfx, closeLine: CLOSE_LINE })(t);
  }, [api, t, pfx]);
  return <Stage gRef={gRef} defs={t >= 82.5 ? api.logoDefs(pfx) : undefined} label={`Frame from the explainer: ${title}`} className="w-full" />;
}

function StillStory({ api, lines }: { api: SFStoryApi | null; lines: Line[] }) {
  return (
    <ol className="mx-auto grid max-w-7xl gap-16 px-4 py-16 sm:px-8">
      {CHAPTERS.slice(1).map((c, k) => {
        const i = k + 1;
        return (
          <li key={c.t} className="grid gap-8 border-t-8 border-vg-white pt-8 md:grid-cols-[minmax(0,1fr)_minmax(0,24rem)] md:gap-14">
            <div>
              <p className="flex items-center gap-3 text-[15px] font-semibold text-vg-soft">
                <span aria-hidden className="block h-5 w-[6px] bg-vg-red" />
                {pad2(i)} / {pad2(CHAPTERS.length - 1)} · {clock(c.t)}
              </p>
              <h3 className="mt-4 text-vg-h2">{c.title}</h3>
              <ul className="mt-6 grid gap-3">
                {lines.filter((l) => l.ch === i).map((l) => (
                  <li key={l.t0 + l.text} className={"text-[20px] font-bold leading-snug " + toneCls(l.tone, true)}>{l.text}</li>
                ))}
              </ul>
              {i === CHAPTERS.length - 1 && <CloseLinks dark />}
            </div>
            <div className="border-l-[10px] border-vg-red">
              {api ? <Still api={api} t={c.still} idx={i} title={c.title} /> : <div className="aspect-[4/5] bg-vg-panel" />}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/* ───────────────────────── scroll-driven story ───────────────────────── */

export default function ScrollStory() {
  const reduced = useReducedMotion() ?? false;
  const secRef = useRef<HTMLElement>(null);
  const stickyRef = useRef<HTMLDivElement>(null);
  const gRef = useRef<SVGGElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const railBars = useRef<(HTMLSpanElement | null)[]>([]);
  const near = useNear(secRef);
  const wide = useWide();
  const { api, failed } = useStoryApi(near);
  const lines = useMemo(() => (api ? buildLines(api) : []), [api]);
  const [ch, setCh] = useState(1);
  const [li, setLi] = useState(-1);

  const tAtScroll = useCallback(() => {
    const sec = secRef.current, st = stickyRef.current;
    if (!sec || !st) return STORY_START;
    const f = Math.min(1, Math.max(0, -sec.getBoundingClientRect().top / pinRange(sec, st)));
    return uToT(knots, f * total);
  }, []);

  // frame loop: scroll → t (rAF-throttled); draw only when t changes; React state only on chapter/line change
  useEffect(() => {
    if (!api || reduced || !gRef.current) return;
    const draw = makeDrawer(gRef.current, api, { captions: false, idPrefix: "story-", closeLine: CLOSE_LINE });
    let lastT = -1, scheduled = 0, lastCh = -1, lastLi = -2, shown = -1;
    const tick = () => {
      scheduled = 0;
      // the drawn clock glides toward the scroll position instead of jumping with each wheel notch;
      // big jumps (chapter links, fast flings past ~4 s of story) snap so they never replay long stretches
      const target = tAtScroll();
      if (shown < 0 || Math.abs(target - shown) > 4) shown = target;
      else shown += (target - shown) * 0.12;
      if (Math.abs(target - shown) < 0.005) shown = target;
      else scheduled = requestAnimationFrame(tick);
      const t = Math.round(shown * 100) / 100; // 10 ms of story clock is below a visible change
      if (t === lastT) return;
      lastT = t;
      draw(t);
      const c = chapterIndex(t), l = lineAt(lines, t, wide);
      if (c !== lastCh) setCh((lastCh = c));
      if (l !== lastLi) setLi((lastLi = l));
      if (barRef.current) barRef.current.style.transform = `scaleX(${(t - STORY_START) / (STORY_END - STORY_START)})`;
      railBars.current.forEach((el, i) => {
        if (!el) return;
        const a = CHAPTERS[i].t, b = i + 1 < CHAPTERS.length ? CHAPTERS[i + 1].t : STORY_END;
        const f = (t - Math.max(a, STORY_START)) / (b - Math.max(a, STORY_START));
        el.style.transform = `scaleX(${f >= 0 && f < 1 ? f : t >= b && i === CHAPTERS.length - 1 ? 1 : 0})`; // only the current chapter
      });
    };
    const onScroll = () => {
      if (!scheduled) scheduled = requestAnimationFrame(tick);
    };
    tick();
    document.fonts.ready.then(() => {
      lastT = -1; // redraw once Inter is in
      onScroll();
    });
    // after a smooth jump, re-sync once the scroll settles (some browsers coalesce the final scroll event)
    const onSettle = () => {
      lastT = -1;
      onScroll();
    };
    const onJump = () => {
      shown = -1; // after a chapter jump, land exactly on the target frame
      onSettle();
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("scrollend", onSettle);
    window.addEventListener("sf:story-sync", onJump);
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(scheduled);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("scrollend", onSettle);
      window.removeEventListener("sf:story-sync", onJump);
      window.removeEventListener("resize", onScroll);
    };
  }, [api, reduced, lines, tAtScroll, wide]);

  const goTo = (i: number, at?: number) => {
    if (i === 0) {
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
      window.dispatchEvent(new Event("sf:hero-replay"));
      return;
    }
    const sec = secRef.current, st = stickyRef.current;
    if (!sec || !st) return;
    const t = at ?? jumpTime(lines, i, wide);
    const top = sec.getBoundingClientRect().top + window.scrollY + (tToU(knots, t) / total) * pinRange(sec, st);
    window.scrollTo({ top: Math.ceil(top), behavior: reduced ? "auto" : "smooth" });
    // fallback for browsers without scrollend: re-sync after the smooth scroll has had time to finish
    [700, 1500].forEach((ms) => window.setTimeout(() => window.dispatchEvent(new Event("sf:story-sync")), ms));
  };

  // other bands link into a chapter: window.dispatchEvent(new CustomEvent("sf:story-goto", { detail: index })),
  // landing on the chapter's representative frame (its still), or { detail: { i, t } } for a set story time
  // inside that chapter (a frame whose caption matches what the band is pointing at).
  const goToRef = useRef(goTo);
  useEffect(() => {
    goToRef.current = goTo;
  });
  useEffect(() => {
    const on = (e: Event) => {
      const det = (e as CustomEvent).detail as number | { i: number; t?: number };
      const i = Number(typeof det === "object" && det ? det.i : det);
      if (!Number.isInteger(i) || i < 0 || i >= CHAPTERS.length) return;
      const at = typeof det === "object" && det && typeof det.t === "number" ? det.t : undefined;
      const end = i + 1 < CHAPTERS.length ? CHAPTERS[i + 1].t : STORY_END;
      if (reduced) secRef.current?.scrollIntoView({ behavior: "auto" });
      else goToRef.current(i, i > 0 ? (at !== undefined && at >= CHAPTERS[i].t && at < end ? at : CHAPTERS[i].still) : undefined);
    };
    window.addEventListener("sf:story-goto", on);
    return () => window.removeEventListener("sf:story-goto", on);
  }, [reduced]);

  const label = "The explainer, told as you scroll: real ScenarioFlow output for a 20 January 2016 query, and real market data from 2000 to 2016";

  if (reduced) {
    return (
      <section id="story" ref={secRef} aria-label={label} className="bg-vg-black text-vg-white">
        <StillStory api={api} lines={lines} />
      </section>
    );
  }

  const chapter = CHAPTERS[ch];
  const dark = chapter.dark;
  const line = li >= 0 ? lines[li] : null;
  const ink = dark ? "text-vg-white" : "text-vg-black";
  const last = ch === CHAPTERS.length - 1;

  return (
    <section
      id="story"
      ref={secRef}
      aria-label={label}
      className={"relative transition-colors duration-300 " + (dark ? "bg-vg-black" : "bg-vg-white")}
      style={{ height: `calc(${total}svh + 100svh)` }}
    >
      <a href="#explore" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-10 focus:bg-vg-red focus:px-4 focus:py-2 focus:text-vg-white">
        Skip the story
      </a>
      <Transcript lines={lines} />
      <div ref={stickyRef} className="sticky top-14 h-[calc(100svh-56px)] overflow-hidden">
        {/* phones: chapter name, a visible way past the story, and one thin red progress bar */}
        <div className={"flex h-10 items-center justify-between gap-3 px-4 text-[13px] font-semibold lg:hidden " + ink}>
          <span className="truncate">{chapter.title}</span>
          <span className="flex shrink-0 items-center gap-3">
            <span className="tabular-nums opacity-70">{pad2(ch)} / {pad2(CHAPTERS.length - 1)}</span>
            <a href="#explore" className="inline-flex min-h-10 items-center border-l-2 border-vg-red pl-3 underline-offset-4 hover:underline">
              Skip ↓
            </a>
          </span>
        </div>
        <div className={"relative h-[3px] lg:hidden " + (dark ? "bg-white/15" : "bg-black/10")} aria-hidden>
          <span ref={barRef} className="absolute inset-0 origin-left bg-vg-red" style={{ transform: "scaleX(0)" }} />
        </div>

        <div className="mx-auto grid h-[calc(100%-43px)] max-w-7xl grid-rows-[auto_minmax(0,1fr)] gap-4 px-4 pt-3 sm:px-8 short:grid-cols-[auto_minmax(0,1fr)] short:grid-rows-1 short:items-start short:gap-6 short:pt-2 lg:h-full lg:grid-cols-[minmax(0,1fr)_auto] lg:grid-rows-1 lg:items-center lg:gap-14 lg:pt-0">
          {/* stage */}
          <div className="order-1 flex justify-center short:justify-start lg:order-2">
            <div
              className={
                "relative border-l-[10px] border-vg-red short:w-[calc((100svh-56px-43px-16px)*0.8)] lg:w-[min(calc((100svh-56px-48px)*0.8),calc(86svh*0.8),46vw)] " +
                (last ? "w-[min(100%,calc((100svh-330px)*0.8))]" : "w-[min(100%,calc((100svh-230px)*0.8))]")
              }
            >
              <Stage gRef={gRef} defs={api ? api.logoDefs("story-") : undefined} label="Live frame of the explainer; the text beside it narrates the same moment" className="w-full" />
              {!api && (
                <p className="absolute inset-0 grid place-items-center p-6 text-center text-[15px] text-vg-grey">
                  {failed ? "The story could not load. The video in the hero shows it." : "Loading the model’s output…"}
                </p>
              )}
            </div>
          </div>

          {/* text column: the chapter title is the headline; the caption is its lede */}
          <div className={"order-2 flex min-h-0 flex-col lg:order-1 lg:h-full lg:justify-between lg:py-10 " + ink}>
            <div className="hidden short:block lg:block">
              <p className={"flex items-center gap-3 text-[14px] font-semibold " + (dark ? "text-vg-soft" : "text-vg-mute")}>
                <span aria-hidden className="block h-5 w-[6px] bg-vg-red" />
                The explainer, live · {pad2(ch)} / {pad2(CHAPTERS.length - 1)} · {clock(chapter.t)}
              </p>
              <h2 className="mt-4 hidden max-w-[34rem] text-[clamp(2.25rem,3.3vw,3.5rem)] font-extrabold leading-[0.98] tracking-[-0.04em] lg:block">
                {chapter.title}
              </h2>
              <span aria-hidden className={"mt-6 hidden h-2 w-20 lg:block " + (dark ? "bg-vg-white" : "bg-vg-black")} />
            </div>

            <div aria-hidden className="lg:min-h-[8rem]">
              {line && (
                <p key={line.t0 + line.text} className={"sf-cap-in max-w-[32rem] text-[17px] font-semibold leading-snug lg:text-[22px] lg:leading-[1.3] " + toneCls(line.tone, dark)}>
                  {line.text}
                </p>
              )}
            </div>
            {last && <CloseLinks dark={dark} />}

            {/* chapter rail (desktop) */}
            <nav aria-label="Story chapters" className="hidden lg:block">
              <ol className={"border-b " + (dark ? "border-white/15" : "border-black/15")}>
                {CHAPTERS.map((c, i) => {
                  const on = i === ch;
                  return (
                    <li key={c.t} className={"border-t " + (dark ? "border-white/15" : "border-black/15")}>
                      <button
                        type="button"
                        onClick={() => goTo(i)}
                        aria-current={on ? "step" : undefined}
                        className={
                          "group relative flex min-h-9 w-full items-baseline gap-3 py-1.5 pl-4 pr-2 text-left transition-colors " +
                          (on ? (dark ? "text-vg-white" : "text-vg-black") : dark ? "text-vg-grey hover:text-vg-white" : "text-vg-mute hover:text-vg-black")
                        }
                      >
                        <span aria-hidden className={"absolute inset-y-0 left-0 w-[5px] " + (on ? "bg-vg-red" : "bg-transparent group-hover:bg-vg-grey")} />
                        <span className="w-9 shrink-0 text-[12px] tabular-nums opacity-80">{clock(c.t)}</span>
                        <span className={"text-[14px] leading-snug " + (on ? "font-bold" : "font-semibold")}>{c.title}</span>
                        <span aria-hidden className="absolute bottom-0 left-[5px] right-0 h-[2px] overflow-hidden">
                          <span
                            ref={(el) => {
                              railBars.current[i] = el;
                            }}
                            className="block h-full origin-left bg-vg-red"
                            style={{ transform: "scaleX(0)" }}
                          />
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </nav>
          </div>
        </div>
      </div>
    </section>
  );
}
