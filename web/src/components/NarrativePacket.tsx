import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  narrativeDefaultIndex,
  parseInlineStrongLine,
  parseNarrativePacket,
} from "@/demo2Presentation";

function inline(text: string, strongClass = ""): ReactNode {
  return text.split("\n").filter((line) => line.trim()).map((line, i) => (
    <p key={i} className="mt-3 first:mt-0">
      {parseInlineStrongLine(line).map((segment, j) => (
        segment.strong
          ? <strong key={j} className={strongClass}>{segment.text}</strong>
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
      className={"min-h-10 shrink-0 whitespace-nowrap border-2 border-vg-black px-4 text-[13px] font-bold transition-colors "
        + (copied ? "bg-vg-black text-vg-white" : "bg-vg-white text-vg-black hover:bg-vg-black hover:text-vg-white")}
    >
      {copied ? "Copied" : "Copy summary"}
    </button>
  );

  if (!sections.length) {
    // Preserve access to the backend text if its packet structure changes.
    return (
      <div className="mt-4">
        <div className="flex justify-end">{copyButton}</div>
        <div className="mt-4 max-h-[28rem] overflow-y-auto whitespace-pre-wrap border-l-[10px] border-vg-red bg-vg-black p-6 text-[15px] leading-relaxed text-vg-white">{md}</div>
      </div>
    );
  }

  const section = sections[selectedIndex];
  return (
    <div className="mt-5">
      {intro && (
        <div className="max-w-3xl text-[14px] leading-relaxed text-vg-mute">
          {inline(intro, "font-semibold text-vg-black")}
        </div>
      )}
      <div className="mt-6">
        <div className="flex items-end justify-between gap-4">
          <div id="np-fmt-label" className="text-[13px] font-semibold text-vg-mute">
            Narrative format · {sections.length}
          </div>
          {copyButton}
        </div>
        <div className="min-w-0">
          <div role="group" aria-labelledby="np-fmt-label" className="mt-3 flex flex-wrap gap-2">
            {sections.map((item, i) => {
              const active = i === selectedIndex;
              return (
                <button
                  key={item.title}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setSelection({ md, index: i })}
                  className={"min-h-9 border-2 px-3 text-left text-[13px] font-bold transition-colors "
                    + (active ? "border-vg-red bg-vg-red text-vg-white" : "border-vg-black bg-vg-white text-vg-black hover:bg-vg-black hover:text-vg-white")}
                >
                  {item.title.replace(/^\s*\d+[-).:]?\s*/, "")}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      <AnimatePresence initial={false} mode="popLayout">
        <motion.div
          key={`${md}-${selectedIndex}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: shouldReduceMotion ? 0 : 0.17, ease: "easeOut" }}
          className="mt-6 grid gap-0 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]"
        >
          <article className="relative bg-vg-black py-7 pr-6 pl-9 text-vg-white sm:pr-8 sm:pl-11">
            <span aria-hidden className="absolute top-0 bottom-0 left-0 w-2.5 bg-vg-red" />
            <h3 className="text-[22px] font-extrabold tracking-[-0.025em]">Scenario summary</h3>
            <p className="mt-1 text-[13px] text-vg-soft">Grounded in the observed factor moves</p>
            <div className="mt-5 text-[16px] leading-[1.65]">{inline(section.positive, "font-bold text-vg-white")}</div>
          </article>
          <article className="relative border-2 border-t-0 border-vg-rule bg-vg-paper py-7 pr-6 pl-9 text-vg-mute sm:pr-8 sm:pl-11 lg:border-t-2 lg:border-l-0">
            <span aria-hidden className="absolute top-0 bottom-0 left-0 w-2.5 bg-vg-soft" />
            <h3 className="text-[22px] font-extrabold tracking-[-0.025em] text-vg-grey">Contrast narrative: written for a different period</h3>
            <p className="mt-1 text-[13px] text-vg-grey">Describes another historical window; a check that the summary is specific to this one</p>
            <div className="mt-5 text-[14px] leading-[1.65]">
              {section.negative
                ? inline(section.negative, "font-bold text-vg-mute")
                : <span className="italic">No contrast narrative for this format.</span>}
            </div>
          </article>
        </motion.div>
      </AnimatePresence>
      <p className="mt-4 text-[13px] text-vg-mute">
        The contrast narrative was written for a different historical period, so it should not fit this episode. Choose among {sections.length} generated formats above.
      </p>
    </div>
  );
}
