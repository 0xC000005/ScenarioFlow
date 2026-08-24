import { Link } from "react-router-dom";

export function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/85 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-2 px-6">
        <Link to="/" className="shrink-0 text-[17px] font-semibold tracking-tight">
          <span className="sm:hidden">ScenarioFlow</span>
          <span className="hidden sm:inline">ScenarioFlow</span>
        </Link>
        <div className="flex items-center gap-3 text-[15px]">
          <a href="#" title="placeholder" className="rounded-full border border-border px-3.5 py-1.5 text-sm hover:bg-secondary">Paper</a>
          <a href="#" title="placeholder" className="rounded-full border border-border px-3.5 py-1.5 text-sm hover:bg-secondary">GitHub</a>
        </div>
      </div>
    </nav>
  );
}
