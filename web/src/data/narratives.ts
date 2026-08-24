// The recommended-narrative bank, copied VERBATIM from the original Gradio Demo 1
// (NARRATIVE_EXAMPLE_FAMILIES in nl_risk_manager_story_gradio_app.py). Six market-regime
// families, each with short / medium / full ("professional read") variants — so a user can
// either type their own narrative or pick one of these to learn what input the system accepts.
export type Length = "short" | "medium" | "full";
export type NarrativeFamily = { key: string; label: string; variants: Record<Length, string> };

export const NARRATIVE_FAMILIES: NarrativeFamily[] = [
  {
    key: "dollar_liquidity_squeeze",
    label: "Dollar liquidity squeeze",
    variants: {
      short: "Dollar funding is tight: DXY and USDJPY are higher, equities are under pressure, volatility is elevated, BBB credit spreads are wider, and commodities are weaker.",
      medium: "The current tape is a dollar-liquidity squeeze. DXY and USDJPY are higher, equities are under pressure, VIX is elevated, BBB spreads are wider, and commodities are weak as investors reduce risk and seek cash-like dollar exposure.",
      full: "Professional read: global dollar liquidity is becoming scarcer. Stronger DXY and USDJPY, weaker equities, elevated volatility, wider BBB spreads, and softer commodities describe the current cross-asset condition. The main portfolio sensitivities are unhedged foreign assets, emerging-market risk, credit beta, and leveraged carry. This is a conditioning view, not a prescribed future path.",
    },
  },
  {
    key: "safe_haven_risk_off",
    label: "Safe-haven risk-off",
    variants: {
      short: "Risk assets are soft, volatility is up, Treasury yields are lower, gold is bid, and credit is showing stress.",
      medium: "The tape is defensive but not a pure dollar squeeze. Equities are selling off, VIX is firmer, BBB spreads are wider, Treasury yields are lower, and gold is catching a haven bid while the dollar channel is less dominant.",
      full: "Professional read: the condition is a safety bid around an equity-credit drawdown. Transmission runs from weaker equities and firmer volatility into lower-quality credit stress, lower Treasury yields, and gold demand. The important distinction is that gold and duration are confirming defense, while FX is not the only channel carrying the stress.",
    },
  },
  {
    key: "weak_dollar_commodity_bid",
    label: "Weak-dollar commodity bid",
    variants: {
      short: "DXY is heavy, USDJPY is firm, crude and gold are bid, rates are a touch higher, and credit is mostly calm.",
      medium: "The active condition is a weak-dollar commodity repricing. DXY is lower while USDJPY, crude, and gold are higher; Treasury yields are only modestly firmer and BBB credit is quiet.",
      full: "Professional read: this is a weak-dollar hard-asset bid with restrained credit stress. The trigger is a softer DXY alongside a firmer USDJPY cross, stronger crude, and stronger gold. Transmission reaches rates through a small backup, but BBB credit remains anchored, so the scenario is commodity/FX-led rather than broad credit deterioration.",
    },
  },
  {
    key: "rates_tightening_pressure",
    label: "Rates tightening pressure",
    variants: {
      short: "Treasury yields are backing up, the dollar is firm, equities are struggling, and volatility is grinding higher.",
      medium: "The current condition is rates-led tightening pressure. The front end and long end are firmer, DXY is supported, equities are under duration pressure, and volatility is rising without a full credit accident.",
      full: "Professional read: the market is repricing around higher rates and a firmer dollar. Equities are struggling with duration pressure, volatility is grinding higher, and credit is fragile but not the first mover. The key exposure is growth-sensitive equity beta and duration-sensitive carry, not a classic safe-haven liquidation.",
    },
  },
  {
    key: "post_stress_reflation_relief",
    label: "Post-stress reflation relief",
    variants: {
      short: "Equities are rebounding, volatility is compressing, crude is firmer, and credit is healing unevenly.",
      medium: "The tape is post-stress reflation relief. SPX is higher, VIX is lower, crude is participating, BBB credit is improving, and high-grade spreads remain the main unresolved split.",
      full: "Professional read: risk appetite is recovering after stress, but the confirmation is uneven. Equity beta and volatility compression lead the move, crude provides reflation support, and BBB spreads improve. The main ambiguity is high-grade credit basis pressure, so the condition is relief with a credit quality split rather than a clean broad-risk rally.",
    },
  },
  {
    key: "split_credit_quality_stress",
    label: "Split credit-quality stress",
    variants: {
      short: "BBB credit is widening while high-grade behaves differently; risk tone is mixed and the spread signal is the issue.",
      medium: "The active condition is split credit-quality stress. Lower-quality spreads are under pressure, high-grade spreads do not confirm in the same direction, and cross-asset risk signals are mixed rather than one-way.",
      full: "Professional read: this is not a simple risk-on or risk-off state. The key condition is divergence inside credit quality: BBB spreads point to stress while high-grade credit is moving differently. Equities, FX, rates, and commodities provide partial context, but portfolio sensitivity should be framed around credit-quality basis and hedges that assume spread cohorts move together.",
    },
  },
];

// 15 fan factors (FAN_MARKET_CHOICES). [label, market-key]
export const FACTORS: [string, string][] = [
  ["SPX", "SPX"], ["VIX", "VIX"], ["BBB OAS", "BBB_OAS"], ["AAA OAS", "AAA_OAS"],
  ["US 2Y", "US2Y"], ["US 10Y", "US10Y"], ["USD/JPY", "USDJPY"], ["DXY", "DXY"],
  ["Gold", "GOLD"], ["Crude oil", "CRUDE_OIL"], ["IV surface avg", "IV_SURFACE"],
  ["IV ATM 3M", "IV_ATM_3M"], ["IV ATM 1Y", "IV_ATM_1Y"], ["IV OTM put 1Y", "IV_OTM_PUT_1Y"],
  ["IV wing 6M", "IV_WING_6M_K130"],
];

export const DEFAULT_NARRATIVE = NARRATIVE_FAMILIES[0].variants.short;
