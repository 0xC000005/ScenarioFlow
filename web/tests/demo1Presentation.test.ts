import assert from "node:assert/strict";
import test from "node:test";

import {
  presentRelatedPeriods,
  scenarioInputsChanged,
} from "../src/demo1Presentation.ts";

test("uses the same rounded narrative match for the label and bar", () => {
  const rows = presentRelatedPeriods([
    { label: "2005-09-09", weight: 0.42, cosine: 0.836, pathCount: 53 },
    { label: "2015-06-26", weight: 0.28, cosine: 0.844, pathCount: 47 },
  ]);

  assert.equal(rows[0].matchLabel, "0.84");
  assert.equal(rows[1].matchLabel, "0.84");
  assert.equal(rows[0].matchPercent, 84);
  assert.equal(rows[1].matchPercent, 84);
  assert.equal(rows[0].pathShareLabel, "53% of generated paths");
  assert.equal(rows[1].pathShareLabel, "47% of generated paths");
});

test("labels the allocation weight honestly when path counts are unavailable", () => {
  const rows = presentRelatedPeriods([
    { label: "2005-09-09", weight: 0.42, cosine: 0.84 },
    { label: "2015-06-26", weight: 0.28, cosine: 0.80 },
  ]);

  assert.equal(rows[0].pathShareLabel, "60% allocation weight");
  assert.equal(rows[1].pathShareLabel, "40% allocation weight");
});

test("detects when editable inputs no longer match the displayed result", () => {
  const submitted = { windowIndex: 40, narrative: "Dollar liquidity tightens" };

  assert.equal(
    scenarioInputsChanged(
      { windowIndex: 40, narrative: "  Dollar liquidity tightens  " },
      submitted,
    ),
    false,
  );
  assert.equal(
    scenarioInputsChanged(
      { windowIndex: 41, narrative: "Dollar liquidity tightens" },
      submitted,
    ),
    true,
  );
  assert.equal(
    scenarioInputsChanged(
      { windowIndex: 40, narrative: "Risk appetite improves" },
      submitted,
    ),
    true,
  );
});
