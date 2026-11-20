// Typed access to public/story/engine.js (the explainer video's own render code, ported verbatim) and a
// promise-singleton script loader, so StrictMode double mounts and the hero + story never inject a script twice.

export type StoryCaption = { t0: number; t1: number; text: string; tone: "ink" | "red" | "mute"; colour: string };
export type RenderOpts = { captions?: boolean; idPrefix?: string; closeLine?: string; layers?: boolean };
export type SFStoryApi = {
  DUR: number;
  S: Record<"s1" | "s2" | "s3" | "s4" | "s5" | "s6" | "s7" | "s8", number>;
  SIZE: [number, number];
  render: (t: number, opts?: RenderOpts) => string;
  readonly captions: StoryCaption[];
  logoDefs: (prefix?: string) => string;
  ready: (kind: "hero" | "story") => boolean;
};

declare global {
  interface Window {
    SFStory?: SFStoryApi;
    DATA?: { story: string; markets: { name: string; ratio: number }[] };
  }
}

const BASE = "/story/";
// the hero's opening scene needs only the narrative-shift data; the rest (≈2.4 MB) loads when the story nears view
const HERO_FILES = ["data_v2.js", "engine.js"];
const STORY_FILES = ["data_v4.js", "trace_v5.js", "data_v11.js", "data_v15_hist.js", "data_v16.js", "logos_white.js"];

const pending = new Map<string, Promise<void>>();

function loadScript(file: string): Promise<void> {
  const hit = pending.get(file);
  if (hit) return hit;
  const p = new Promise<void>((resolve, reject) => {
    const el = document.createElement("script");
    el.src = BASE + file;
    el.async = false; // keep execution order (data before engine)
    el.onload = () => resolve();
    el.onerror = () => {
      pending.delete(file);
      el.remove();
      reject(new Error(`could not load ${file}`));
    };
    document.head.appendChild(el);
  });
  pending.set(file, p);
  return p;
}

export function loadHero(): Promise<SFStoryApi> {
  return Promise.all(HERO_FILES.map(loadScript)).then(() => {
    if (!window.SFStory) throw new Error("story engine missing");
    return window.SFStory;
  });
}

export function loadStory(): Promise<SFStoryApi> {
  return Promise.all([...HERO_FILES, ...STORY_FILES].map(loadScript)).then(() => {
    if (!window.SFStory || !window.SFStory.ready("story")) throw new Error("story data missing");
    return window.SFStory;
  });
}

let prefetched = false;
/** Download the story files ahead of time without executing them (the scripts run later from cache). */
export function prefetchStory() {
  if (prefetched) return;
  prefetched = true;
  for (const f of STORY_FILES) {
    if (pending.has(f)) continue;
    const l = document.createElement("link");
    l.rel = "prefetch";
    l.as = "script";
    l.href = BASE + f;
    document.head.appendChild(l);
  }
}
