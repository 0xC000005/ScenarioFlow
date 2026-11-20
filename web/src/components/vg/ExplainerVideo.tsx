import { useEffect, useRef, useState } from "react";
import "@/components/vg/story/story.css";

// The explainer video (1080×1350, 89 s, no audio), offered as a file for readers who prefer to watch it.
// The landing page itself renders the same scenes live from the same code and data (see story/ScrollStory).

const SRC = "/video/scenarioflow_explainer.mp4";

/** "Watch the 90-second video" link that opens the mp4 in a native modal dialog with controls. */
export default function ExplainerVideo({ className = "", label = "Watch the 90-second video" }: { className?: string; label?: string }) {
  const dlg = useRef<HTMLDialogElement>(null);
  const vid = useRef<HTMLVideoElement>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const d = dlg.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      vid.current?.play().catch(() => undefined);
    } else if (!open && d.open) {
      d.close();
    }
  }, [open]);

  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className={className}>
        <svg width="12" height="14" viewBox="0 0 12 14" aria-hidden className="shrink-0">
          <path d="M1 1l10 6-10 6z" fill="currentColor" />
        </svg>
        {label}
      </button>
      <dialog
        ref={dlg}
        className="sf-video m-auto w-[min(92vw,calc(86svh*0.8))] max-w-none bg-vg-black p-0 text-vg-white"
        aria-label="ScenarioFlow explainer video, 89 seconds, no audio"
        onClose={() => {
          setOpen(false);
          vid.current?.pause();
        }}
        onClick={(e) => {
          if (e.target === e.currentTarget) setOpen(false); // click on the backdrop
        }}
      >
        <div className="flex items-center justify-between gap-4 border-b-4 border-vg-red px-4 py-2">
          <p className="text-[14px] font-semibold">The explainer, 89 seconds, no sound</p>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="inline-flex min-h-11 items-center border-2 border-vg-white px-4 text-[14px] font-semibold transition-colors hover:bg-vg-white hover:text-vg-black"
          >
            Close
          </button>
        </div>
        {open && (
          <video
            ref={vid}
            className="block aspect-[4/5] w-full bg-vg-panel"
            src={SRC}
            poster="/video/explainer_poster.png"
            controls
            playsInline
            muted
            preload="metadata"
          />
        )}
      </dialog>
    </>
  );
}
