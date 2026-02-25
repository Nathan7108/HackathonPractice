"use client";

import { useId } from "react";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

type KpiCardProps = {
  value: string | number;
  label: string;
  trend?: { direction: "up" | "down"; delta: string };
  sparklineData?: number[];
};

export function KpiCard({
  value,
  label,
  trend,
  sparklineData = [],
}: KpiCardProps) {
  const gradientId = useId();
  const data = sparklineData.map((v, i) => ({ x: i, v }));
  const trendUp = trend?.direction === "up";
  const trendDown = trend?.direction === "down";

  return (
    <Card className="min-w-[180px] flex-1">
      <CardHeader className="pb-1 pt-4 px-4">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0">
        <div className="flex items-end justify-between gap-2">
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tabular-nums">{value}</span>
            {trend && (
              <span
                className={`text-xs font-medium ${
                  trendUp ? "text-red-600" : trendDown ? "text-green-600" : "text-muted-foreground"
                }`}
              >
                {trendUp ? "↑" : trendDown ? "↓" : ""}
                {trend.delta}
              </span>
            )}
          </div>
          {data.length > 0 && (
            <div className="h-8 w-16 shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke="hsl(var(--primary))"
                    fill={`url(#${gradientId})`}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
