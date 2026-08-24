export type NarrativeSection = {
  title: string;
  positive: string;
  negative: string;
};

export type InlineStrongSegment = {
  text: string;
  strong: boolean;
};

export function parseInlineStrongLine(line: string): InlineStrongSegment[] {
  return line
    .split(/(\*\*[^*]+\*\*)/g)
    .filter(Boolean)
    .map((segment) => {
      const strong = segment.startsWith("**") && segment.endsWith("**");
      return {
        text: strong ? segment.slice(2, -2) : segment,
        strong,
      };
    });
}

export type Demo2ResultIdentity = {
  requestId: number;
  submittedDate: string;
};

export type CoherentDemo2Snapshot<
  Scenario extends Demo2ResultIdentity,
  Narrative extends Demo2ResultIdentity,
> = {
  scenario: Scenario;
  narrative: Narrative;
};

export function captureCoherentDemo2Snapshot<
  Scenario extends Demo2ResultIdentity,
  Narrative extends Demo2ResultIdentity,
>(
  scenario: Scenario | null,
  narrative: Narrative | null,
): CoherentDemo2Snapshot<Scenario, Narrative> | null {
  if (
    !scenario
    || !narrative
    || scenario.requestId !== narrative.requestId
    || scenario.submittedDate !== narrative.submittedDate
  ) {
    return null;
  }
  return { scenario, narrative };
}

export function restoreDemo2FailureResults<
  Scenario extends Demo2ResultIdentity,
  Narrative extends Demo2ResultIdentity,
>(
  previous: CoherentDemo2Snapshot<Scenario, Narrative> | null,
  publishedScenario: Scenario | null,
): { scenario: Scenario | null; narrative: Narrative | null } {
  return previous ?? { scenario: publishedScenario, narrative: null };
}

export function parseNarrativePacket(md: string): {
  intro: string;
  sections: NarrativeSection[];
} {
  const bySection = md.split(/\n###\s+/);
  const intro = bySection[0].replace(/^#{1,3}\s+.*$/m, "").trim();
  const sections = bySection
    .slice(1)
    .map((block) => {
      const newline = block.indexOf("\n");
      const title = (newline >= 0 ? block.slice(0, newline) : block).trim();
      const body = newline >= 0 ? block.slice(newline) : "";
      const parts = body.split(/\*\*Hard[- ]negative\*\*/i);
      const positive = parts[0].replace(/\*\*Positive\*\*/i, "").trim();
      const negative = (parts[1] ?? "").trim();
      return { title, positive, negative };
    })
    .filter((section) => section.positive || section.negative);

  return { intro, sections };
}

export function narrativeDefaultIndex(sections: NarrativeSection[]): number {
  const memoIndex = sections.findIndex((section) =>
    section.title.replace(/^\s*\d+[-).:]?\s*/, "").trim().toLowerCase() ===
    "risk manager memo"
  );
  return memoIndex >= 0 ? memoIndex : 0;
}

export function resultDateChanged(
  selectedDate: string,
  submittedDate: string | null,
): boolean {
  return submittedDate !== null && selectedDate !== submittedDate;
}

export function demo2InputsChanged(
  selectedDate: string,
  activeSubmittedDate: string | null,
  displayedResultDates: Array<string | null>,
): boolean {
  if (activeSubmittedDate !== null) {
    return selectedDate !== activeSubmittedDate;
  }
  return displayedResultDates.some((submittedDate) =>
    resultDateChanged(selectedDate, submittedDate),
  );
}

export function isCurrentDemo2Request(
  responseRequestId: number,
  currentRequestId: number,
): boolean {
  return responseRequestId === currentRequestId;
}

export function demo2DownloadFilename(submittedDate: string): string {
  return `narrative_${submittedDate}.md`;
}

export function directionArrow(direction: string): "↑" | "↓" | "→" {
  const normalized = direction.toLowerCase();
  if (normalized === "up" || normalized === "higher") return "↑";
  if (normalized === "down" || normalized === "lower") return "↓";
  return "→";
}
