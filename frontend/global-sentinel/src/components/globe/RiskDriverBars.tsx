"use client";

import type { FeatureImportance } from "@/lib/types";

type Props = { drivers: FeatureImportance[] };

export function RiskDriverBars({ drivers }: Props) {
  const max = Math.max(...drivers.map((d) => d.percentage), 1);
  return (
    <div className="space-y-2">
      {drivers.map((d) => (
        <div key={d.name} className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="truncate">{d.name}</span>
            <span className="text-muted-foreground tabular-nums">{d.percentage}%</span>
          </div>
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${(d.percentage / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
