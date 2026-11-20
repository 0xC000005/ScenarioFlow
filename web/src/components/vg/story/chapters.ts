// Chapters of the explainer (start times are the video's scene starts, S in engine.js).
// `pace` = scroll height per story-second (in viewport-height %), slower through the dense scenes.
// `still` = the representative frame shown under reduced motion (after the scene's fades, before beat fade-outs).
// `dark` = the scene's own background, so the text column flips with it.
// `beats` = extra timed lines for scenes whose video has no caption strip; each is text that the frame
// itself draws (or its plain restatement), never a new claim. `echo` marks a beat that repeats text the frame
// draws legibly: on wide screens it is left out of the caption column (one voice per screen), on phones, where
// the frame's own labels are too small to read, it is shown; the screen-reader transcript always keeps it.

export type Beat = { t0: number; text: string; tone: "ink" | "red" | "mute"; echo?: boolean };
export type Chapter = { t: number; title: string; still: number; pace: number; dark: boolean; beats: Beat[] };

export const STORY_START = 7.75; // scene 2 with "Risk is discussed in words," fully drawn (opaque at S.s2 + 0.7)
export const HERO_END = 6.85; // last frame before scene 2 begins to blend in
export const CLOSE_LINE = "Live demos, paper and poster below."; // the video's card says "in the post"

export const CHAPTERS: Chapter[] = [
  { t: 0, title: "Describe the market in words", still: HERO_END, pace: 0, dark: true, beats: [] },
  {
    t: 7.0, title: "The gap", still: 14.6, pace: 9, dark: false,
    beats: [
      { t0: 7.0, text: "Risk is discussed in words, but simulated in equations whose form is fixed by hand.", tone: "ink" },
      { t0: 11.3, text: "Today’s tools prescribe the dynamics, or replay history.", tone: "ink" },
    ],
  },
  {
    t: 15.5, title: "Words find past markets", still: 24.6, pace: 10, dark: false,
    beats: [
      { t0: 15.5, text: "Your words are matched to past markets by similarity; those periods reshape the range.", tone: "ink" },
      { t0: 16.2, text: "Crude oil, next 30 days, from market state only.", tone: "mute", echo: true },
      { t0: 18.4, text: "+ your words: crude’s 10–90% range on day 30 is ×{CRUDE} wider.", tone: "red", echo: true },
    ],
  },
  { t: 25.0, title: "Every market at once", still: 32.6, pace: 9, dark: true, beats: [] },
  { t: 33.0, title: "Inside the network", still: 46.6, pace: 15, dark: true, beats: [] },
  {
    t: 47.0, title: "What the model remembers", still: 66.6, pace: 12, dark: true,
    beats: [
      { t0: 47.4, text: "Each dot is one past market, as the model remembers it.", tone: "ink" },
      { t0: 54.6, text: "{HIT} of a stressed day’s 10 closest memories are stressed too, where {SHARE}% would be chance. Nobody labelled them.", tone: "red", echo: true },
    ],
  },
  {
    t: 67.0, title: "How it draws the future", still: 82.0, pace: 14, dark: false,
    beats: [
      { t0: 78.3, text: "Stocks down, volatility up.", tone: "ink", echo: true },
      { t0: 79.9, text: "Nobody wrote that rule down. The model learned most of it.", tone: "red", echo: true },
    ],
  },
  {
    // the card draws the paper title; the column carries only the links (the title is set once more in band 11)
    t: 82.5, title: "Live demos, paper and poster", still: 88.6, pace: 6, dark: true, beats: [],
  },
];

export const STORY_END = 89;

export const chapterIndex = (t: number) => {
  let i = 0;
  CHAPTERS.forEach((c, k) => {
    if (t >= c.t - 1e-6) i = k;
  });
  return i;
};

// piecewise-linear scroll map over the story's range [STORY_START, STORY_END]
type Knot = { t: number; u: number };
// PACE_SCALE stretches every chapter (one wheel notch ≈ half a story-second); HOLD is scroll (svh) spent on a
// chapter's finished frame before the next scene starts to blend in, so each scene can be read at rest.
const FADE = 0.6, FADE_PACE = 0.8, PACE_SCALE = 2.4, HOLD = 45, HOLD_LEAD = 0.4;
export function buildKnots(): { knots: Knot[]; total: number } {
  const knots: Knot[] = [{ t: STORY_START, u: 0 }];
  let u = 0;
  for (let i = 1; i < CHAPTERS.length; i++) {
    const a = Math.max(CHAPTERS[i].t, STORY_START);
    const b = i + 1 < CHAPTERS.length ? CHAPTERS[i + 1].t : STORY_END;
    const pace = CHAPTERS[i].pace * PACE_SCALE;
    // hold on the chapter's clean `still` frame: scenes crossfade over ±0.14 s around the next scene's start
    // (the red wipe into "Inside the network" over ±0.35 s), so the pause must sit before that window
    const hb = i + 1 < CHAPTERS.length ? Math.min(CHAPTERS[i].still, b - HOLD_LEAD) : b;
    if (a > STORY_START && b - a > 2 * FADE) {
      u += FADE * pace * FADE_PACE;
      knots.push({ t: a + FADE, u });
      u += (hb - a - FADE) * pace;
    } else {
      u += (hb - a) * pace;
    }
    knots.push({ t: hb, u });
    u += HOLD;
    knots.push({ t: hb, u });
    if (hb < b) {
      u += (b - hb) * pace;
      knots.push({ t: b, u });
    }
  }
  return { knots, total: u };
}

export function uToT(knots: Knot[], u: number) {
  if (u <= 0) return knots[0].t;
  for (let i = 1; i < knots.length; i++) {
    const a = knots[i - 1], b = knots[i];
    if (u <= b.u) return a.t + ((u - a.u) / (b.u - a.u)) * (b.t - a.t);
  }
  return knots[knots.length - 1].t;
}

export function tToU(knots: Knot[], t: number) {
  if (t <= knots[0].t) return 0;
  for (let i = 1; i < knots.length; i++) {
    const a = knots[i - 1], b = knots[i];
    if (t <= b.t) return a.u + ((t - a.t) / (b.t - a.t)) * (b.u - a.u);
  }
  return knots[knots.length - 1].u;
}
