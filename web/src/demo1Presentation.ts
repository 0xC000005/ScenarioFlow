export type RelatedPeriodInput = {
  label: string;
  weight: number;
  cosine: number;
  pathCount?: number;
};

export type RelatedPeriodPresentation = RelatedPeriodInput & {
  matchLabel: string;
  matchPercent: number;
  pathShareLabel: string;
};

export type ScenarioInputs = {
  windowIndex: number;
  narrative: string;
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function presentRelatedPeriods(
  periods: RelatedPeriodInput[],
): RelatedPeriodPresentation[] {
  const totalPaths = periods.reduce(
    (total, period) => total + Math.max(0, period.pathCount ?? 0),
    0,
  );
  const totalWeight = periods.reduce(
    (total, period) => total + Math.max(0, period.weight),
    0,
  );

  return periods.map((period) => {
    const matchPercent = Math.round(clamp(period.cosine, 0, 1) * 100);
    const hasPathCounts = totalPaths > 0;
    const share = hasPathCounts
      ? Math.round((Math.max(0, period.pathCount ?? 0) / totalPaths) * 100)
      : totalWeight > 0
        ? Math.round((Math.max(0, period.weight) / totalWeight) * 100)
        : 0;

    return {
      ...period,
      matchLabel: (matchPercent / 100).toFixed(2),
      matchPercent,
      pathShareLabel: hasPathCounts
        ? `${share}% of generated paths`
        : `${share}% allocation weight`,
    };
  });
}

export function scenarioInputsChanged(
  current: ScenarioInputs,
  submitted: ScenarioInputs | null,
): boolean {
  if (!submitted) return false;
  return current.windowIndex !== submitted.windowIndex
    || current.narrative.trim() !== submitted.narrative.trim();
}
