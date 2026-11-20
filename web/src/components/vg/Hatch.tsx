import { VG } from "@/lib/palette";

// The explainer's 45° hatching: grey = market state only, red = + your words.
// Pattern ids are global in a document, so each SVG passes a unique `id` prefix.
export function HatchDefs({ id, scale = 1 }: { id: string; scale?: number }) {
  const s = 9 * scale;
  return (
    <defs>
      <pattern id={`${id}-red`} width={s} height={s} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width={3.2 * scale} height={s} fill={VG.red} />
      </pattern>
      <pattern id={`${id}-grey`} width={s} height={s} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width={2.2 * scale} height={s} fill={VG.grey} />
      </pattern>
      <pattern id={`${id}-white`} width={s} height={s} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width={2.2 * scale} height={s} fill="#9A9A9A" />
      </pattern>
    </defs>
  );
}

// eslint-disable-next-line react-refresh/only-export-components -- tiny helper paired with HatchDefs
export const hatch = (id: string, tone: "red" | "grey" | "white") => `url(#${id}-${tone})`;
