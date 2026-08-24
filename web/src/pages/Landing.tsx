import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { motion, useInView, useReducedMotion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { FactorGrid } from "@/components/FactorGrid";
import PipelineSchematic from "@/components/PipelineSchematic";

function WorkflowComparison() {
  const imageRef = useRef<HTMLVideoElement>(null);
  const marketRef = useRef<HTMLVideoElement>(null);
  const comparisonRef = useRef<HTMLDivElement>(null);
  const endedVideos = useRef(new Set<HTMLVideoElement>());
  const inView = useInView(comparisonRef, { amount: 0.25 });
  const reduced = useReducedMotion() ?? false;

  useEffect(() => {
    const videos = [imageRef.current, marketRef.current].filter((video): video is HTMLVideoElement => Boolean(video));
    if (reduced) {
      videos.forEach((video) => {
        video.pause();
        if (Number.isFinite(video.duration)) video.currentTime = Math.max(0, video.duration - 0.04);
      });
      return;
    }
    if (!inView) {
      videos.forEach((video) => video.pause());
      return;
    }
    videos.forEach((video) => {
      if (!endedVideos.current.has(video)) void video.play().catch(() => undefined);
    });
  }, [inView, reduced]);

  const handleEnded = (video: HTMLVideoElement) => {
    endedVideos.current.add(video);
  };

  const replay = () => {
    if (reduced) return;
    const videos = [imageRef.current, marketRef.current].filter((video): video is HTMLVideoElement => Boolean(video));
    videos.forEach((video) => {
      video.pause();
      video.currentTime = 0;
    });
    endedVideos.current.clear();
    if (inView) videos.forEach((video) => void video.play().catch(() => undefined));
  };

  const videoProps = {
    muted: true,
    playsInline: true,
    preload: "auto" as const,
    className: "w-full rounded-lg border border-border bg-white",
  };

  return (
    <div ref={comparisonRef}>
      <div className="mt-10 grid gap-6 md:grid-cols-2">
        <Card className="flex items-center justify-center p-4 sm:p-6">
          <video
            ref={imageRef}
            src="/anim/image_gen.mp4"
            {...videoProps}
            onEnded={(event) => handleEnded(event.currentTarget)}
            onLoadedMetadata={(event) => {
              if (reduced) event.currentTarget.currentTime = Math.max(0, event.currentTarget.duration - 0.04);
            }}
          />
        </Card>
        <Card className="flex items-center justify-center p-4 sm:p-6">
          <video
            ref={marketRef}
            src="/anim/market_gen.mp4"
            {...videoProps}
            onEnded={(event) => handleEnded(event.currentTarget)}
            onLoadedMetadata={(event) => {
              if (reduced) event.currentTarget.currentTime = Math.max(0, event.currentTarget.duration - 0.04);
            }}
          />
        </Card>
      </div>
      <div className="mt-3 flex justify-center">
        <button
          type="button"
          onClick={replay}
          disabled={reduced}
          className="inline-flex min-h-9 items-center rounded-full border border-border bg-background/70 px-3 text-xs font-medium text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-default disabled:opacity-45"
          aria-label="Replay image and market generation comparison"
        >
          ↻&nbsp; Replay comparison
        </button>
      </div>
    </div>
  );
}

function PlotLegend() {
  const items = [
    { label: "History", color: "#1d1d1f", width: 1.5, dashed: false },
    { label: "Generated paths", color: "#0071e3", width: 0.8, dashed: false },
    { label: "Median", color: "#0071e3", width: 2, dashed: false },
    { label: "Realized path", color: "#1d1d1f", width: 1.4, dashed: true },
  ];
  return (
    <div className="mb-5 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs text-muted-foreground" aria-label="Chart legend">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <svg width="25" height="8" viewBox="0 0 25 8" aria-hidden>
            <line x1="1" y1="4" x2="24" y2="4" stroke={item.color} strokeWidth={item.width} strokeDasharray={item.dashed ? "3 2" : undefined} />
          </svg>
          {item.label}
        </span>
      ))}
    </div>
  );
}

export default function Landing() {
  const reduced = useReducedMotion() ?? false;
  const heroInitial = reduced ? false : { opacity: 0, y: 8 };
  const heroTransition = reduced ? { duration: 0 } : { duration: 0.22, ease: "easeOut" as const };

  return (
    <main>
      {/* HERO — foundation, video analogy */}
      <section className="mx-auto max-w-6xl px-6 pb-20 pt-28 text-center">
        <motion.h1 initial={heroInitial} animate={{ opacity: 1, y: 0 }} transition={heroTransition} className="mx-auto mt-4 max-w-3xl text-[2.5rem] font-semibold leading-[1.06] tracking-tight sm:text-5xl">
          ScenarioFlow: A Temporal Foundation Model with Multimodal Market Conditioning
        </motion.h1>
        <motion.p initial={heroInitial} animate={{ opacity: 1, y: 0 }} transition={{ ...heroTransition, delay: reduced ? 0 : 0.035 }} className="mx-auto mt-5 max-w-2xl text-xl leading-relaxed text-muted-foreground">
          Given numerical history or a natural-language description, ScenarioFlow generates a conditional distribution
          of joint, multi-day paths across many variables. The resulting scenarios achieve state-of-the-art fidelity and
          realism.
        </motion.p>
        <motion.div initial={heroInitial} animate={{ opacity: 1, y: 0 }} transition={{ ...heroTransition, delay: reduced ? 0 : 0.07 }} className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link to="/demo1" className="flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--primary)_40%,var(--border))] bg-[color-mix(in_oklab,var(--primary)_8%,white)] px-5 py-3 text-[17px] font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--primary)_15%,white)]">
            <span className="rounded-full bg-[color-mix(in_oklab,var(--primary)_12%,white)] px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-primary">Demo</span>
            Narrative → Scenario
          </Link>
          <Link to="/demo2" className="flex items-center gap-2 rounded-full border border-[color-mix(in_oklab,var(--primary)_40%,var(--border))] bg-[color-mix(in_oklab,var(--primary)_8%,white)] px-5 py-3 text-[17px] font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--primary)_15%,white)]">
            <span className="rounded-full bg-[color-mix(in_oklab,var(--primary)_12%,white)] px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-primary">Demo</span>
            Scenario → Narrative
          </Link>
        </motion.div>
      </section>

      {/* FOUNDATION — generating scenarios */}
      <section className="border-y border-border bg-secondary/40 py-20">
        <div className="mx-auto max-w-6xl px-6">
          <h2 className="text-center text-3xl font-semibold tracking-tight">From Images to Market Time Series</h2>
          <p className="mx-auto mt-3 max-w-3xl text-center text-lg leading-relaxed text-muted-foreground">
            Nano Banana generates images, Sora generates videos, and ScenarioFlow applies the same generative idea to
            markets. A market snapshot is like a frame, and a multi-day scenario is a sequence of connected frames.
          </p>
          <WorkflowComparison />
          <p className="mx-auto mt-5 max-w-3xl text-center text-base leading-relaxed text-muted-foreground">
            Autoregressive conditional flow matching transforms Gaussian noise into the next joint market state and
            repeats this process to build a complete time series. Through self-supervised training, ScenarioFlow learns
            how markets move together across variables and through time, rather than prescribing separate dynamics asset
            by asset.
          </p>

          <div className="mt-12">
            <div className="text-center text-xl font-semibold">One Joint Scenario Distribution Across Markets</div>
            <p className="mx-auto mt-1.5 mb-6 max-w-2xl text-center text-[15px] leading-relaxed text-muted-foreground">
              The panels show equities, rates, credit, volatility, FX, and commodities evolving together within the same
              generated scenario.
            </p>
            <PlotLegend />
            <Card className="p-4 sm:p-6"><FactorGrid src="/data/panel_startonly.json" /></Card>
          </div>
        </div>
      </section>

      {/* APPLICATIONS — what you can do with it */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <p className="text-center text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Built on the foundation</p>
        <h2 className="mt-3 text-center text-3xl font-semibold tracking-tight">What you can do with it</h2>
        <p className="mx-auto mt-3 max-w-2xl text-center text-lg text-muted-foreground">
          Generate market scenarios from language, or summarize an observed market episode.
        </p>

        <div className="mt-10 grid gap-6">
          <Card className="flex flex-col p-6">
            <div className="text-lg font-semibold">Narrative → Scenario</div>
            <p className="mt-1.5 text-[15px] leading-relaxed text-muted-foreground">
              Describe a market condition and generate its conditional distribution.
            </p>
            <div className="my-5"><PipelineSchematic direction="narrative-to-scenario" /></div>
            <motion.div whileTap={reduced ? undefined : { opacity: 0.72 }} transition={{ duration: 0.18 }} className="mt-auto">
              <Link to="/demo1" className="group inline-flex min-h-11 items-center rounded-lg px-1 text-[15px] font-medium text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary">
                Open the demo <span className="ml-1 transition-transform duration-[180ms] group-hover:translate-x-[3px] group-focus-visible:translate-x-[3px]">›</span>
              </Link>
            </motion.div>
          </Card>
          <Card className="flex flex-col p-6">
            <div className="text-lg font-semibold">Scenario → Narrative</div>
            <p className="mt-1.5 text-[15px] leading-relaxed text-muted-foreground">
              Choose an observed market episode and summarize its movements in plain language.
            </p>
            <div className="my-5"><PipelineSchematic direction="scenario-to-narrative" /></div>
            <motion.div whileTap={reduced ? undefined : { opacity: 0.72 }} transition={{ duration: 0.18 }} className="mt-auto">
              <Link to="/demo2" className="group inline-flex min-h-11 items-center rounded-lg px-1 text-[15px] font-medium text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary">
                Open the demo <span className="ml-1 transition-transform duration-[180ms] group-hover:translate-x-[3px] group-focus-visible:translate-x-[3px]">›</span>
              </Link>
            </motion.div>
          </Card>
        </div>
      </section>

      <footer className="border-t border-border py-10">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 text-sm text-muted-foreground">
          <span>ScenarioFlow</span>
          <span className="flex gap-5">
            <a href="#" title="placeholder" className="hover:text-foreground">Paper</a>
            <a href="#" title="placeholder" className="hover:text-foreground">GitHub</a>
          </span>
        </div>
      </footer>
    </main>
  );
}
