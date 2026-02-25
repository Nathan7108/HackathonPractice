"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { RISK_DISTRIBUTION } from "@/lib/dashboard-data";

const TIER_COLORS: Record<string, string> = {
  LOW: "hsl(142, 76%, 36%)",
  MODERATE: "hsl(48, 96%, 53%)",
  ELEVATED: "hsl(25, 95%, 53%)",
  HIGH: "hsl(0, 84%, 60%)",
  CRITICAL: "hsl(0, 72%, 51%)",
};

export function RiskDistributionChart() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Risk Distribution</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={RISK_DISTRIBUTION}
              margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
              layout="vertical"
            >
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="tier"
                width={80}
                tick={{ fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={28}>
                {RISK_DISTRIBUTION.map((entry, index) => (
                  <Cell
                    key={entry.tier}
                    fill={TIER_COLORS[entry.tier] ?? "hsl(var(--muted))"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
