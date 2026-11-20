import { Link } from "react-router-dom";

const linkCls = "inline-flex min-h-11 items-center text-[15px] font-semibold text-vg-white transition-colors hover:text-vg-red";

export default function Footer() {
  return (
    <footer className="bg-vg-black text-vg-white">
      <div className="h-2 bg-vg-red" aria-hidden />
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-14 sm:px-8 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <Link to="/" className="inline-flex items-center gap-2.5 text-[28px] font-extrabold tracking-[-0.04em]">
            <span aria-hidden className="block h-7 w-2.5 bg-vg-red" />
            ScenarioFlow
          </Link>
          <p className="mt-4 max-w-sm text-[14px] leading-relaxed text-vg-grey">
            Conditional distributions of joint market paths, from recent market history and, optionally, a plain-language description.
          </p>
        </div>
        <div>
          <h2 className="text-[13px] font-semibold text-vg-grey">Live demos</h2>
          <ul className="mt-2">
            <li><Link to="/demo1" className={linkCls}>Narrative → Scenario</Link></li>
            <li><Link to="/demo2" className={linkCls}>Scenario → Narrative</Link></li>
          </ul>
        </div>
        <div>
          <h2 className="text-[13px] font-semibold text-vg-grey">Research</h2>
          <ul className="mt-2">
            <li>
              <a href="/paper/ScenarioFlow_FMTS2026_paper.pdf" target="_blank" rel="noopener" className={linkCls}>
                Paper (PDF)
              </a>
            </li>
            <li>
              <a href="/poster/ScenarioFlow_FMTS2026_poster_61x85cm.pdf" target="_blank" rel="noopener" className={linkCls}>
                Poster (PDF)
              </a>
            </li>
            <li>
              <a href="/video/scenarioflow_explainer.mp4" target="_blank" rel="noopener" className={linkCls}>
                Explainer video (MP4, 89 s)
              </a>
            </li>
            <li>
              <a href="https://osf.io/ywpc5/" target="_blank" rel="noopener" className={linkCls}>
                Code and artifacts (OSF)
              </a>
            </li>
            <li>
              <a href="https://github.com/0xC000005/ScenarioFlow" target="_blank" rel="noopener" className={linkCls}>
                Code on GitHub
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        <div className="border-t border-white/15 py-5 text-[13px] text-vg-soft">FMTS @ NeurIPS 2026 workshop</div>
      </div>
    </footer>
  );
}
