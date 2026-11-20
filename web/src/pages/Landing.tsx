import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import ScenarioToNarrative from "@/components/vg/ScenarioToNarrative";
import HeroCrude from "@/components/vg/landing/HeroCrude";
import BeforeAfterGrid from "@/components/vg/landing/BeforeAfterGrid";
import EmbeddingMap from "@/components/vg/landing/EmbeddingMap";
import ScrollStory from "@/components/vg/story/ScrollStory";
import Footer from "@/components/vg/Footer";

const POSTER_PDF = "/poster/ScenarioFlow_FMTS2026_poster_61x85cm.pdf";
const PAPER_PDF = "/paper/ScenarioFlow_FMTS2026_paper.pdf"; // the camera-ready version
const heroLink = "inline-flex min-h-11 items-center text-vg-white underline decoration-vg-red decoration-2 underline-offset-[6px] transition-colors hover:text-vg-red";

const btnRed =
  "inline-flex min-h-12 items-center bg-vg-red px-6 text-[16px] font-semibold text-vg-white transition-colors hover:bg-vg-red-deep";
const btnOutlineWhite =
  "inline-flex min-h-12 items-center border-2 border-vg-white px-6 text-[16px] font-semibold text-vg-white transition-colors hover:bg-vg-white hover:text-vg-black";
const btnBlack =
  "inline-flex min-h-12 items-center bg-vg-black px-6 text-[16px] font-semibold text-vg-white transition-colors hover:bg-vg-red";

/**
 * The constructivist wedge from the explainer's closing card: a red right triangle cut into a band's top-right
 * corner. One per band, the site's single recurring diagonal outside the story stage.
 */
function Wedge({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={"pointer-events-none absolute right-0 top-0 block h-[clamp(48px,11vw,168px)] w-[clamp(72px,16vw,248px)] bg-vg-red " + className}
      style={{ clipPath: "polygon(0 0, 100% 0, 100% 100%)" }}
    />
  );
}

/** Short red vertical bar: the accent mark used for every section kicker. */
function Kicker({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return (
    <p className={"flex items-center gap-3 text-[15px] font-semibold " + (dark ? "text-vg-soft" : "text-vg-mute")}>
      <span aria-hidden className="block h-5 w-[6px] bg-vg-red" />
      {children}
    </p>
  );
}

function Hero() {
  const reduced = useReducedMotion() ?? false;
  const line = (delay: number) =>
    reduced
      ? { initial: false as const }
      : {
          initial: { y: "105%" },
          animate: { y: 0 },
          transition: { duration: 0.75, delay, ease: [0.2, 0.75, 0.15, 1] as const },
        };
  const fade = (delay: number) =>
    reduced
      ? { initial: false as const }
      : { initial: { opacity: 0 }, animate: { opacity: 1 }, transition: { duration: 0.4, delay } };

  return (
    <section className="bg-vg-black text-vg-white" aria-labelledby="hero-h">
      <div className="mx-auto grid max-w-7xl gap-12 px-4 pb-12 pt-12 sm:px-8 md:pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:items-center lg:gap-14 lg:pb-16 lg:pt-20">
        <div className="flex flex-col">
          <h1 id="hero-h" className="text-vg-h1 lg:text-[clamp(3.5rem,5.5vw,5.75rem)]">
            <span className="block overflow-hidden pb-[0.06em]">
              <motion.span className="block" {...line(0.05)}>Describe the market in words.</motion.span>
            </span>
            <span className="block overflow-hidden pb-[0.06em]">
              <motion.span className="block text-vg-red" {...line(0.22)}>See its future distribution.</motion.span>
            </span>
          </h1>
          <motion.div {...fade(0.55)}>
            <div aria-hidden className="mt-8 h-2 w-24 bg-vg-white" />
            <p className="mt-8 max-w-[34rem] text-[19px] leading-relaxed text-vg-soft">
              Give ScenarioFlow a market&rsquo;s recent history, and optionally a plain-language description of the
              conditions, and it generates a conditional distribution of joint, multi-day paths across equities, rates,
              credit, volatility, FX and commodities.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link to="/demo1" className={btnRed}>Narrative → Scenario</Link>
              <Link to="/demo2" className={btnOutlineWhite}>Scenario → Narrative</Link>
            </div>
            {/* paper and poster up front, so nobody has to scroll to the end for them */}
            <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-[15px] font-semibold">
              <span className="text-vg-grey">FMTS @ NeurIPS 2026 workshop</span>
              <a href={PAPER_PDF} target="_blank" rel="noopener" className={heroLink}>Paper (PDF)</a>
              <a href={POSTER_PDF} target="_blank" rel="noopener" className={heroLink}>Poster (PDF)</a>
              <a href="https://github.com/0xC000005/ScenarioFlow" target="_blank" rel="noopener" className={heroLink}>Code (GitHub)</a>
            </div>
          </motion.div>
        </div>
        <motion.div {...fade(0.35)}>
          <HeroCrude />
        </motion.div>
      </div>
      <div className="mx-auto max-w-7xl px-4 pb-10 sm:px-8">
        <div className="flex items-center gap-4 border-t-2 border-white/20 pt-4 text-[14px] font-semibold text-vg-soft">
          <span aria-hidden className="block h-5 w-[6px] bg-vg-red" />
          <span>Keep scrolling: the explainer plays as you scroll, drawn live from the same model output.</span>
          <svg width="14" height="18" viewBox="0 0 14 18" aria-hidden className="ml-auto shrink-0 text-vg-red">
            <path d="M7 0v15M1 10l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2.5" />
          </svg>
        </div>
      </div>
    </section>
  );
}

function WorkflowBand({
  id,
  dark,
  paper = false,
  kicker,
  title,
  body,
  to,
  cta,
  children,
}: {
  id?: string;
  dark: boolean;
  paper?: boolean;
  kicker: string;
  title: ReactNode;
  body: ReactNode;
  to: string;
  cta: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className={"relative scroll-mt-14 overflow-hidden " + (dark ? "bg-vg-black text-vg-white" : paper ? "bg-vg-paper text-vg-black" : "bg-vg-white text-vg-black")}>
      <Wedge />
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-8 md:py-24">
        <div className={"border-t-8 pt-8 " + (dark ? "border-vg-white" : "border-vg-black")}>
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:gap-16">
            <div>
              <Kicker dark={dark}>{kicker}</Kicker>
              <h2 className="mt-4 text-vg-h2">{title}</h2>
            </div>
            <div className="lg:pt-10">
              <p className={"max-w-xl text-[18px] leading-relaxed " + (dark ? "text-vg-soft" : "text-vg-mute")}>{body}</p>
              <Link to={to} className={"mt-7 " + (dark ? btnRed : btnBlack)}>
                {cta}
              </Link>
            </div>
          </div>
          <div className="mt-12">{children}</div>
        </div>
      </div>
    </section>
  );
}

function PaperPoster() {
  return (
    <section id="paper" className="relative scroll-mt-14 overflow-hidden bg-vg-paper text-vg-black" aria-labelledby="paper-h">
      <Wedge className="hidden md:block" />
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-16 sm:px-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] md:gap-16 md:py-24">
        <a
          href={POSTER_PDF}
          target="_blank"
          rel="noopener"
          className="group block border-[6px] border-vg-black bg-vg-white"
          aria-label="Open the conference poster (PDF, new tab)"
        >
          <img
            src="/poster/poster_preview.png"
            alt="ScenarioFlow conference poster for FMTS at NeurIPS 2026"
            width={807}
            height={1124}
            loading="lazy"
            className="block aspect-[807/1124] h-auto w-full transition-opacity group-hover:opacity-90"
          />
        </a>
        <div className="flex flex-col md:pt-4">
          <Kicker>Paper and poster</Kicker>
          <h2 id="paper-h" className="mt-4 text-vg-h2">
            ScenarioFlow: multimodal market conditioning for heterogeneous multivariate time-series simulation
          </h2>
          <div className="mt-8 flex">
            <span aria-hidden className="w-[10px] shrink-0 bg-vg-red" />
            <p className="bg-vg-black px-4 py-3 text-[16px] font-semibold text-vg-white">FMTS @ NeurIPS 2026 workshop</p>
          </div>
          <p className="mt-8 max-w-xl text-[18px] leading-relaxed text-vg-mute">
            The paper describes the generator behind both demos: one model that conditions on recent market history, and
            optionally on a plain-language description of the conditions, and returns a distribution of joint paths rather
            than a single forecast.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <a
              href={PAPER_PDF}
              target="_blank"
              rel="noopener"
              className="inline-flex min-h-12 items-center bg-vg-black px-6 text-[16px] font-semibold text-vg-white transition-colors hover:bg-vg-red"
            >
              Read the paper
            </a>
            <a
              href={POSTER_PDF}
              target="_blank"
              rel="noopener"
              className="inline-flex min-h-12 items-center border-2 border-vg-black px-6 text-[16px] font-semibold transition-colors hover:bg-vg-black hover:text-vg-white"
            >
              Poster PDF
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Landing() {
  return (
    <main>
      <Hero />
      <ScrollStory />

      <WorkflowBand
        id="explore"
        dark={false}
        kicker="Before and after your words · Narrative → Scenario"
        title="Same market state. Add the words. Every range changes."
        body={
          <>
            Grey is each market&rsquo;s 30-day range from market state alone; red is the same state plus the words of the
            20 January 2016 query. The words widen some ranges and narrow others, by drawing on related past periods.
            They change the conditional distribution of outcomes; they are not a forecast of where a market goes.
          </>
        }
        to="/demo1"
        cta="Try your own words"
      >
        <BeforeAfterGrid />
      </WorkflowBand>

      <WorkflowBand
        id="map"
        paper
        dark={false}
        kicker="Where your words land"
        title="Your words become a point among 4,010 past markets."
        body={
          <>
            The model keeps a memory of every past 30-day market. Your words are placed on the same map, and the closest
            past markets are the ones it draws on: the same three dates as above. It is the map from the story&rsquo;s
            chapter{" "}
            <a
              href="#story"
              onClick={(e) => {
                e.preventDefault();
                window.dispatchEvent(new CustomEvent("sf:story-goto", { detail: { i: 5, t: 59.6 } }));
              }}
              className="font-semibold text-vg-black underline decoration-vg-red decoration-2 underline-offset-4 hover:text-vg-red"
            >
              &ldquo;What the model remembers&rdquo;
            </a>
            .
          </>
        }
        to="/demo1"
        cta="Open the narrative demo"
      >
        <EmbeddingMap />
      </WorkflowBand>

      <WorkflowBand
        dark
        kicker="The other direction · Scenario → Narrative"
        title="Paths in. A plain-language account out."
        body={
          <>
            Choose an observed 30-day market episode. A language model reads the joint moves across markets and writes
            them up in plain language, so a scenario can be discussed in the words a risk committee already uses.
          </>
        }
        to="/demo2"
        cta="Open the episode demo"
      >
        <ScenarioToNarrative />
      </WorkflowBand>

      <PaperPoster />
      <Footer />
    </main>
  );
}
