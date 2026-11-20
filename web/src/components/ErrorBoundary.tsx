import { Component, type ReactNode } from "react";

// Catches render-time crashes so a single error shows a clean message instead of a white screen.
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error) { console.error("[boundary]", error); }
  render() {
    if (this.state.error) {
      return (
        <main className="min-h-[70vh] bg-vg-black text-vg-white">
          <div className="mx-auto max-w-7xl px-4 py-20 sm:px-8 sm:py-28">
            <span aria-hidden className="block h-10 w-2.5 bg-vg-red" />
            <h1 className="mt-6 max-w-3xl text-vg-h1">Something went wrong rendering this view.</h1>
            <p className="mt-5 max-w-2xl border-l-2 border-vg-mute pl-4 text-[15px] leading-relaxed text-vg-soft">
              {String(this.state.error.message || this.state.error)}
            </p>
            <button
              type="button"
              onClick={() => { this.setState({ error: null }); location.reload(); }}
              className="mt-8 min-h-12 bg-vg-red px-7 text-[15px] font-bold text-vg-white transition-colors hover:bg-vg-red-deep"
            >
              Reload
            </button>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}
