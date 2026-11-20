import { clamp01 } from "./timeline";

/**
 * Text that types out with progress p (0..1). The untyped remainder stays in the layout so nothing
 * reflows while typing; screen readers get the whole text. By default the remainder is a dim
 * "ghost" of the real text (so a panel is never empty before it plays); pass ghost="" to hide it.
 */
export function Typed({ text, p, caret = true, ghost = "text-transparent" }: { text: string; p: number; caret?: boolean; ghost?: string }) {
  const n = Math.round(clamp01(p) * text.length);
  const typing = p > 0 && p < 1;
  return (
    <>
      <span>{text.slice(0, n)}</span>
      {caret && typing && <span aria-hidden className="inline-block h-[1em] w-[0.5em] translate-y-[0.15em] bg-vg-red" />}
      <span className={ghost || "text-transparent"}>
        {text.slice(n)}
      </span>
    </>
  );
}

/** Square, outlined Replay control. */
export function ReplayButton({ onClick, label = "Replay", dark = false }: { onClick: () => void; label?: string; dark?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "inline-flex h-9 items-center gap-2 border-2 px-3 text-[14px] font-semibold transition-colors " +
        (dark
          ? "border-vg-white text-vg-white hover:bg-vg-white hover:text-vg-black"
          : "border-vg-black text-vg-black hover:bg-vg-black hover:text-vg-white")
      }
    >
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
        <path d="M2 7a5 5 0 1 0 1.6-3.7" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M1.2 1.2v3.6h3.6" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      {label}
    </button>
  );
}

/** Shown in place of a figure whose data file failed to load: a short message and a retry. */
export function FigureLoadError({ onRetry, minHeight = 240 }: { onRetry: () => void; minHeight?: number }) {
  return (
    <div role="alert" className="flex flex-col items-start justify-center gap-3 border-l-[6px] border-vg-red bg-vg-paper p-6" style={{ minHeight }}>
      <p className="text-[17px] font-extrabold tracking-[-0.01em] text-vg-black">Figure data could not load.</p>
      <p className="max-w-md text-[14px] leading-relaxed text-vg-mute">This figure is drawn from a stored model output file. Check the connection, then retry.</p>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-10 items-center border-2 border-vg-black px-4 text-[14px] font-bold transition-colors hover:bg-vg-black hover:text-vg-white"
      >
        Retry
      </button>
    </div>
  );
}
