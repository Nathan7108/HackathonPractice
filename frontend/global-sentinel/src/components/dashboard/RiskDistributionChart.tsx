"use client";

import { Card, CardContent } from "@/components/ui/card";
import { PanelHeader } from "@/components/dashboard/PanelHeader";
import { RISK_DISTRIBUTION } from "@/lib/dashboard-data";

const TIER_COLORS: Record<string, string> = {
  LOW: "#22c55e",
  MODERATE: "#eab308",
  ELEVATED: "#f97316",
  HIGH: "#dc2626",
  CRITICAL: "#7f1d1d",
};

const ORDER: Array<keyof typeof TIER_COLORS> = [
  "LOW",
  "MODERATE",
  "ELEVATED",
  "HIGH",
  "CRITICAL",
];

export function RiskDistributionChart() {
  const total = RISK_DISTRIBUTION.reduce((s, e) => s + e.count, 0);
  const segments = ORDER.map((tier) => {
    const entry = RISK_DISTRIBUTION.find((e) => e.tier === tier);
    const count = entry?.count ?? 0;
    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
    return { tier, count, pct, color: TIER_COLORS[tier] };
  }).filter((s) => s.pct > 0);

  return (
    <Card className="p-0 border border-border rounded-sm shadow-none h-full min-h-0 flex flex-col">
      <PanelHeader title="Risk Distribution" />
      <CardContent className="p-3 flex-1 min-h-0">
        {/* Single horizontal stacked bar with percentage labels ON the bar */}
        <div className="flex h-8 w-full rounded-sm overflow-hidden bg-gray-100">
          {segments.map(({ tier, pct, color }) => (
            <div
              key={tier}
              className="h-full flex items-center justify-center transition-[width]"
              style={{
                width: `${pct}%`,
                minWidth: pct > 0 ? "2px" : 0,
                backgroundColor: color,
              }}
              title={`${tier}: ${pct}%`}
            >
              {pct >= 8 && (
                <span className="text-[10px] font-semibold text-white drop-shadow-[0_0_1px_rgba(0,0,0,0.8)] truncate px-0.5">
                  {pct}%
                </span>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
