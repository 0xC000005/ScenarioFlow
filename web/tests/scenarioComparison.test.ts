import assert from "node:assert/strict";
import test from "node:test";

import {
  COMPARISON_MARKETS,
  PAPER_TONE_COLORS,
  buildScenarioComparisonRows,
  comparisonPresentation,
  directionPresentation,
} from "../src/scenarioComparison.ts";

const report = {
  generation: {
    terminal_delta_summary: [
      {
        market: "BBB_OAS",
        mean_terminal_delta: -0.01,
        p10: -0.04,
        p90: 0.02,
        terminal_mean_sigma: -0.3,
        terminal_probability_up: 0.4,
        terminal_probability_down: 0.6,
      },
      {
        market: "SPX",
        mean_terminal_delta: 17,
        p10: -10,
        p90: 40,
        terminal_mean_sigma: 0.6,
        terminal_probability_up: 0.73,
        terminal_probability_down: 0.27,
      },
      {
        market: "IV_ATM_3M",
        mean_terminal_delta: 0.01,
        p10: -0.01,
        p90: 0.03,
        terminal_probability_up: 0.7,
        terminal_probability_down: 0.3,
      },
    ],
    start_only_baseline: {
      terminal_delta_summary: [
        {
          market: "BBB_OAS",
          mean_terminal_delta: -0.066,
          p10: -0.09,
          p90: -0.03,
          terminal_mean_sigma: -1.6,
          terminal_probability_up: 0.05,
          terminal_probability_down: 0.95,
        },
        {
          market: "SPX",
          mean_terminal_delta: -4,
          p10: -14,
          p90: 6,
          terminal_mean_sigma: -0.2,
          terminal_probability_up: 0.45,
          terminal_probability_down: 0.55,
        },
      ],
    },
  },
};

test("uses the same eleven-market order as paper Table 13", () => {
  assert.deepEqual(
    COMPARISON_MARKETS.map(({ key }) => key),
    [
      "IV_SURFACE",
      "SPX",
      "US2Y",
      "US10Y",
      "BBB_OAS",
      "AAA_OAS",
      "USDJPY",
      "DXY",
      "GOLD",
      "CRUDE_OIL",
      "VIX",
    ],
  );
});

test("builds the Table 13 direction, share, move, and comparison fields", () => {
  const rows = buildScenarioComparisonRows(report);

  assert.deepEqual(rows.map(({ marketKey }) => marketKey), ["SPX", "BBB_OAS"]);
  assert.deepEqual(rows[0], {
    marketKey: "SPX",
    marketLabel: "SPX",
    startOnly: {
      direction: "Mixed",
      majorityShare: "55% down",
      meanMove: "-4 pts / -0.2σ",
    },
    narrative: {
      direction: "Up",
      majorityShare: "73% up",
      meanMove: "+17 pts / +0.6σ",
    },
    comparison: "More up than baseline",
  });
  assert.equal(rows[1].marketLabel, "BBB OAS");
  assert.deepEqual(rows[1].startOnly, {
    direction: "Down",
    majorityShare: "95% down",
    meanMove: "-6.6 bp / -1.6σ",
  });
  assert.deepEqual(rows[1].narrative, {
    direction: "Down",
    majorityShare: "60% down",
    meanMove: "-1.0 bp / -0.3σ",
  });
  assert.equal(rows[1].comparison, "Less down than baseline");
});

test("matches the paper arrows and color roles", () => {
  assert.deepEqual(PAPER_TONE_COLORS, {
    green: "#16803A",
    red: "#C53030",
    gray: "#5F6368",
    amber: "#B7791F",
  });
  assert.deepEqual(directionPresentation("Up"), { arrow: "↑", tone: "green" });
  assert.deepEqual(directionPresentation("Down"), { arrow: "↓", tone: "red" });
  assert.deepEqual(directionPresentation("Mixed"), { arrow: "", tone: "gray" });
  assert.deepEqual(comparisonPresentation("More up than baseline"), {
    arrow: "↑",
    tone: "green",
  });
  assert.deepEqual(comparisonPresentation("More down than baseline"), {
    arrow: "↓",
    tone: "red",
  });
  assert.deepEqual(comparisonPresentation("Less down than baseline"), {
    arrow: "↓",
    tone: "amber",
  });
  assert.deepEqual(comparisonPresentation("Similar"), {
    arrow: "—",
    tone: "gray",
  });
});

test("uses the paper's short Similar label", () => {
  const similar = buildScenarioComparisonRows({
    generation: {
      terminal_delta_summary: [
        {
          market: "US10Y",
          mean_terminal_delta: -0.079,
          p10: -0.12,
          p90: -0.03,
          terminal_probability_up: 0.2,
          terminal_probability_down: 0.8,
        },
      ],
      start_only_baseline: {
        terminal_delta_summary: [
          {
            market: "US10Y",
            mean_terminal_delta: -0.088,
            p10: -0.13,
            p90: -0.04,
            terminal_probability_up: 0.17,
            terminal_probability_down: 0.83,
          },
        ],
      },
    },
  });
  assert.equal(similar[0].comparison, "Similar");
});

test("matches Python's half-even percentage formatting", () => {
  const rows = buildScenarioComparisonRows({
    generation: {
      terminal_delta_summary: [
        {
          market: "DXY",
          mean_terminal_delta: 0.2,
          p10: -0.2,
          p90: 0.6,
          terminal_probability_up: 0.625,
          terminal_probability_down: 0.375,
        },
      ],
      start_only_baseline: { terminal_delta_summary: [] },
    },
  });
  assert.equal(rows[0].narrative.majorityShare, "62% up");
});
