import assert from "node:assert/strict";
import test from "node:test";

import {
  captureCoherentDemo2Snapshot,
  demo2DownloadFilename,
  demo2InputsChanged,
  directionArrow,
  isCurrentDemo2Request,
  narrativeDefaultIndex,
  parseInlineStrongLine,
  parseNarrativePacket,
  restoreDemo2FailureResults,
  resultDateChanged,
} from "../src/demo2Presentation.ts";

const packet = `# Scenario narrative packet

Generated from the selected episode.

### 1. Market Brief
**Positive**
Equities fell while volatility rose.

**Hard-negative**
Equities rallied while volatility eased.

### 2. Risk Manager Memo
**Positive**
Risk appetite deteriorated across the episode.

**Hard negative**
Risk appetite improved across the episode.`;

test("parses packet sections and chooses Risk Manager Memo by title", () => {
  const parsed = parseNarrativePacket(packet);

  assert.equal(parsed.intro, "Generated from the selected episode.");
  assert.deepEqual(parsed.sections, [
    {
      title: "1. Market Brief",
      positive: "Equities fell while volatility rose.",
      negative: "Equities rallied while volatility eased.",
    },
    {
      title: "2. Risk Manager Memo",
      positive: "Risk appetite deteriorated across the episode.",
      negative: "Risk appetite improved across the episode.",
    },
  ]);
  assert.equal(narrativeDefaultIndex(parsed.sections), 1);
});

test("parses inline bold labels without exposing Markdown markers", () => {
  assert.deepEqual(
    parseInlineStrongLine(
      "The **Positive** narrative is faithful; the **Hard negative** is contradictory.",
    ),
    [
      { text: "The ", strong: false },
      { text: "Positive", strong: true },
      { text: " narrative is faithful; the ", strong: false },
      { text: "Hard negative", strong: true },
      { text: " is contradictory.", strong: false },
    ],
  );
});

test("falls back to the first narrative format when the memo is absent", () => {
  assert.equal(
    narrativeDefaultIndex([
      { title: "Market Brief", positive: "A", negative: "B" },
      { title: "Portfolio Note", positive: "C", negative: "D" },
    ]),
    0,
  );
});

test("compares the editable date with the submitted result date", () => {
  assert.equal(resultDateChanged("2008-09-22", "2008-09-22"), false);
  assert.equal(resultDateChanged("2008-10-01", "2008-09-22"), true);
  assert.equal(resultDateChanged("2008-10-01", null), false);
});

test("accepts async results only from the latest request", () => {
  assert.equal(isCurrentDemo2Request(4, 4), true);
  assert.equal(isCurrentDemo2Request(3, 4), false);
});

test("builds downloads from the submitted date and uses neutral arrow glyphs", () => {
  assert.equal(demo2DownloadFilename("2008-09-22"), "narrative_2008-09-22.md");
  assert.equal(directionArrow("up"), "↑");
  assert.equal(directionArrow("Higher"), "↑");
  assert.equal(directionArrow("down"), "↓");
  assert.equal(directionArrow("lower"), "↓");
  assert.equal(directionArrow("mixed"), "→");
});

test("restores the coherent result pair captured before a stage-two failure", () => {
  const previousScenario = {
    requestId: 4,
    submittedDate: "2008-09-22",
    rows: ["prior scenario"],
  };
  const previousNarrative = {
    requestId: 4,
    submittedDate: "2008-09-22",
    markdown: "prior narrative",
  };
  const publishedScenario = {
    requestId: 5,
    submittedDate: "2008-10-01",
    rows: ["new scenario"],
  };

  const captured = captureCoherentDemo2Snapshot(
    previousScenario,
    previousNarrative,
  );

  assert.deepEqual(
    restoreDemo2FailureResults(captured, publishedScenario),
    { scenario: previousScenario, narrative: previousNarrative },
  );
});

test("keeps a published scenario without inventing a narrative when no prior pair exists", () => {
  const publishedScenario = {
    requestId: 1,
    submittedDate: "2008-09-22",
    rows: ["new scenario"],
  };

  assert.deepEqual(
    restoreDemo2FailureResults(
      captureCoherentDemo2Snapshot(null, null),
      publishedScenario,
    ),
    { scenario: publishedScenario, narrative: null },
  );
  assert.deepEqual(
    restoreDemo2FailureResults(captureCoherentDemo2Snapshot(null, null), null),
    { scenario: null, narrative: null },
  );
});

test("does not report drift from an old narrative during a same-date active request", () => {
  assert.equal(
    demo2InputsChanged(
      "2008-10-01",
      "2008-10-01",
      ["2008-10-01", "2008-09-22"],
    ),
    false,
  );
});

test("reports drift when the editable date changes during an active request", () => {
  assert.equal(
    demo2InputsChanged(
      "2008-10-15",
      "2008-10-01",
      ["2008-10-01", "2008-09-22"],
    ),
    true,
  );
});

test("reports drift from displayed results while idle", () => {
  assert.equal(
    demo2InputsChanged(
      "2008-10-01",
      null,
      ["2008-09-22", "2008-09-22"],
    ),
    true,
  );
});
