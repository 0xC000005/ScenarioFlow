import type { RenderOpts, SFStoryApi } from "./engine";

declare global {
  interface Window {
    /** set to [] in the console (or a test) to record per-frame build/apply cost, in ms */
    __sfPerf?: { t: number; build: number; apply: number; bytes: number }[];
  }
}

const OPEN = "<!--L-->", CLOSE = "<!--/L-->";
type Seg = { html: string };

/**
 * Split the frame at the engine's layer markers into balanced, top-level fragments (in draw order).
 */
function segments(s: string): Seg[] {
  if (!s.includes(OPEN)) return [{ html: s }];
  const out: Seg[] = [];
  let i = 0;
  for (;;) {
    const a = s.indexOf(OPEN, i);
    if (a < 0) break;
    const b = s.indexOf(CLOSE, a);
    if (b < 0) break;
    out.push({ html: s.slice(i, a) }, { html: s.slice(a + OPEN.length, b) });
    i = b + CLOSE.length;
  }
  out.push({ html: s.slice(i) });
  return out;
}

/**
 * Returns draw(t): renders frame t of the story into the <g>. The frame is applied as a few sibling groups in
 * draw order; a group whose markup did not change since the last frame keeps its DOM (the 4,010-dot memory map
 * and the 1,500-day cloud are static for most of their scenes), so nothing drawn changes, only what is re-parsed.
 */
export function makeDrawer(root: SVGGElement, api: SFStoryApi, opts: RenderOpts) {
  const NS = "http://www.w3.org/2000/svg";
  let last: Seg[] = [];
  root.innerHTML = "";
  return (t: number) => {
    const perf = window.__sfPerf;
    const a = perf ? performance.now() : 0;
    const segs = segments(api.render(t, { ...opts, layers: true }));
    const b = perf ? performance.now() : 0;
    while (root.childNodes.length < segs.length) root.appendChild(document.createElementNS(NS, "g"));
    for (let i = 0; i < root.childNodes.length; i++) {
      const html = segs[i]?.html ?? "";
      if (html !== last[i]?.html) (root.childNodes[i] as SVGGElement).innerHTML = html;
    }
    last = segs;
    if (perf) {
      root.getBBox(); // force style + layout so the apply cost includes parse and layout
      perf.push({ t, build: b - a, apply: performance.now() - b, bytes: segs.reduce((n, x) => n + x.html.length, 0) });
    }
  };
}
