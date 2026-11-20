import type { ReactNode } from "react";

/**
 * Shared control-column parts for both demo pages, so the two workbenches use one vocabulary:
 * rule-topped numbered sections (red numeral + bold label) and one red call-to-action.
 */

/** A numbered control section: a heavy rule on top, red numeral, bold label, then its controls. */
export function ControlStep({
  n,
  label,
  htmlFor,
  first = false,
  children,
}: {
  n: number;
  label: ReactNode;
  htmlFor?: string;
  first?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className={first ? "border-t-8 border-vg-black pt-6" : "mt-8 border-t-2 border-vg-black pt-5"}>
      <div className="flex items-baseline gap-3">
        <span aria-hidden="true" className="text-[28px] font-extrabold leading-none tracking-[-0.04em] text-vg-red">{n}</span>
        {htmlFor
          ? <label htmlFor={htmlFor} className="text-[17px] font-extrabold tracking-[-0.01em]">{label}</label>
          : <span className="text-[17px] font-extrabold tracking-[-0.01em]">{label}</span>}
      </div>
      {children}
    </div>
  );
}

/** The demos' one primary action: full-width red bar, left-aligned label, white end-bar (or elapsed seconds while running). */
export function GenerateButton({
  onClick,
  disabled,
  busy,
  label,
  busyLabel,
  elapsed,
}: {
  onClick: () => void;
  disabled?: boolean;
  busy: boolean;
  label: string;
  busyLabel: string;
  elapsed?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="mt-8 flex min-h-14 w-full items-center justify-between gap-4 bg-vg-red px-5 py-3 text-left text-[17px] font-extrabold tracking-[-0.01em] text-vg-white transition-colors hover:bg-vg-red-deep disabled:cursor-not-allowed disabled:bg-vg-mute"
    >
      <span>{busy ? busyLabel : label}</span>
      {busy && elapsed != null
        ? <span className="text-[15px] font-bold tabular-nums text-vg-white/85">{Math.floor(elapsed)} s</span>
        : <span aria-hidden="true" className="block h-6 w-2 bg-vg-white" />}
    </button>
  );
}
