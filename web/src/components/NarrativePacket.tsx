import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  narrativeDefaultIndex,
  parseInlineStrongLine,
  parseNarrativePacket,
} from "@/demo2Presentation";

function inline(text: string): ReactNode {
  return text.split("\n").filter((line) => line.trim()).map((line, i) => (
    <p key={i} className="mt-2 first:mt-0">
      {parseInlineStrongLine(line).map((segment, j) => (
        segment.strong
          ? <strong key={j}>{segment.text}</strong>
          : segment.text
      ))}
    </p>
  ));
}

export function NarrativePacketView({ md }: { md: string }) {
  const shouldReduceMotion = useReducedMotion() ?? false;
  const { intro, sections } = parseNarrativePacket(md);
  const defaultIndex = narrativeDefaultIndex(sections);
  const [selection, setSelection] = useState({ md, index: defaultIndex });
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const copyReset = useRef<number | null>(null);
  const selectedIndex = selection.md === md
    ? Math.min(selection.index, Math.max(sections.length - 1, 0))
    : defaultIndex;
  const copyKey = sections.length ? `${md}:${selectedIndex}` : md;
  const copied = copiedKey === copyKey;

  useEffect(() => () => {
    if (copyReset.current !== null) window.clearTimeout(copyReset.current);
  }, []);

  async function copySummary() {
    try {
      const summary = sections.length ? sections[selectedIndex].positive : md;
      await navigator.clipboard.writeText(summary);
      setCopiedKey(copyKey);
      if (copyReset.current !== null) window.clearTimeout(copyReset.current);
      copyReset.current = window.setTimeout(() => {
        setCopiedKey((current) => current === copyKey ? null : current);
      }, 1600);
    } catch (error) {
      console.error("Unable to copy narrative summary:", error);
    }
  }

  const copyButton = (
    <button
      type="button"
      onClick={copySummary}
      aria-live="polite"
      className="min-h-9 rounded-full border border-border px-3 text-xs font-medium transition-colors hover:bg-secondary"
    >
      {copied ? "Copied" : "Copy summary"}
    </button>
  );

  if (!sections.length) {
    // Preserve access to the backend text if its packet structure changes.
    return (
      <div className="mt-3">
        <div className="flex justify-end">{copyButton}</div>
        <div className="mt-3 max-h-[28rem] overflow-y-auto whitespace-pre-wrap text-[14px] leading-relaxed">{md}</div>
      </div>
    );
  }

  const section = sections[selectedIndex];
  return (
    <div className="mt-3">
      {intro && (
        <div className="text-xs leading-relaxed text-muted-foreground">
          {inline(intro)}
        </div>
      )}
      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
        <div className="min-w-0 flex-1">
          <label htmlFor="np-fmt" className="text-xs font-medium text-muted-foreground">Narrative format</label>
          <select
            id="np-fmt"
            value={selectedIndex}
            onChange={(event) => setSelection({ md, index: Number(event.target.value) })}
            className="mt-1 w-full rounded-lg border border-border bg-background p-2 text-sm"
          >
            {sections.map((item, i) => <option key={item.title} value={i}>{item.title}</option>)}
          </select>
        </div>
        <div className="self-end">{copyButton}</div>
      </div>
      <AnimatePresence initial={false} mode="popLayout">
        <motion.div
          key={`${md}-${selectedIndex}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: shouldReduceMotion ? 0 : 0.17, ease: "easeOut" }}
          className="mt-4 grid gap-3 lg:grid-cols-2"
        >
          <div className="rounded-lg border border-border p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-foreground">Scenario summary</div>
            <div className="mt-2 text-[14px] leading-relaxed text-foreground">{inline(section.positive)}</div>
          </div>
          <div className="rounded-lg border border-border/70 bg-muted/30 p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Contradictory control</div>
            <div className="mt-2 text-[13px] leading-relaxed text-muted-foreground">
              {section.negative
                ? inline(section.negative)
                : <span className="italic">No contradictory control for this format.</span>}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
      <p className="mt-3 text-xs text-muted-foreground">
        The control intentionally conflicts with the episode. Choose among {sections.length} generated formats above.
      </p>
    </div>
  );
}
