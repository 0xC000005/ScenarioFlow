// Site palette for SVG/Plotly code that cannot read CSS variables.
// Keep in sync with the --vg-* tokens in index.css (same values as the explainer video).
export const VG = {
  black: "#000000",
  white: "#FFFFFF",
  red: "#E81132",
  redDeep: "#B80D27",
  grey: "#8C8C8C",
  mute: "#5C5C5C",
  soft: "#BDBDBD",
  rule: "#DADADA",
  panel: "#1A1A1A",
  paper: "#F4F4F4",
} as const;

export const VG_FONT = "'Inter Variable', Inter, system-ui, sans-serif";

// rgba helpers for Plotly fills (Plotly cannot use SVG patterns, so bands use flat red/grey alpha)
export const redA = (a: number) => `rgba(232,17,50,${a})`;
export const greyA = (a: number) => `rgba(140,140,140,${a})`;
