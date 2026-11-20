import { Link, NavLink } from "react-router-dom";

const item = ({ isActive }: { isActive: boolean }) =>
  "inline-flex min-h-11 items-center whitespace-nowrap px-3 text-[14px] font-semibold transition-colors sm:text-[15px] " +
  (isActive ? "bg-vg-red text-vg-white" : "text-vg-white hover:text-vg-red");

export function Nav() {
  return (
    <nav className="sticky top-0 z-50 bg-vg-black text-vg-white">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-2 px-4 sm:px-8">
        <Link to="/" className="flex shrink-0 items-center gap-2 text-[17px] font-extrabold sm:gap-2.5 sm:text-[19px] tracking-[-0.03em]">
          <span aria-hidden className="block h-5 w-2 bg-vg-red" />
          ScenarioFlow
        </Link>
        <div className="flex items-center gap-0 sm:gap-1">
          {/* wide screens: paper and poster always one click away (phones get them under the hero buttons) */}
          <a href="/paper/ScenarioFlow_FMTS2026_paper.pdf" target="_blank" rel="noopener"
            className="hidden min-h-11 items-center px-3 text-[15px] font-semibold text-vg-soft transition-colors hover:text-vg-red lg:inline-flex">Paper</a>
          <a href="/poster/ScenarioFlow_FMTS2026_poster_61x85cm.pdf" target="_blank" rel="noopener"
            className="hidden min-h-11 items-center px-3 text-[15px] font-semibold text-vg-soft transition-colors hover:text-vg-red lg:inline-flex">Poster</a>
          <a href="https://github.com/0xC000005/ScenarioFlow" target="_blank" rel="noopener"
            className="mr-3 hidden min-h-11 items-center px-3 text-[15px] font-semibold text-vg-soft transition-colors hover:text-vg-red lg:inline-flex">GitHub</a>
          <NavLink to="/demo1" className={item}>
            {/* phones: the full name, stacked on two lines so it fits beside the logo */}
            <span className="flex flex-col text-[12px] leading-[1.15] sm:hidden"><span>Narrative →</span><span>Scenario</span></span>
            <span className="hidden sm:inline">Narrative → Scenario</span>
          </NavLink>
          <NavLink to="/demo2" className={item}>
            <span className="flex flex-col text-[12px] leading-[1.15] sm:hidden"><span>Scenario →</span><span>Narrative</span></span>
            <span className="hidden sm:inline">Scenario → Narrative</span>
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
