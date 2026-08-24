import { Component, type ReactNode } from "react";

// Catches render-time crashes so a single error shows a clean message instead of a white screen.
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error) { console.error("[boundary]", error); }
  render() {
    if (this.state.error) {
      return (
        <main className="mx-auto max-w-xl px-6 py-24 text-center">
          <h1 className="text-2xl font-semibold">Something went wrong rendering this view.</h1>
          <p className="mt-2 text-sm text-muted-foreground">{String(this.state.error.message || this.state.error)}</p>
          <button onClick={() => { this.setState({ error: null }); location.reload(); }}
            className="mt-5 rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground">Reload</button>
        </main>
      );
    }
    return this.props.children;
  }
}
