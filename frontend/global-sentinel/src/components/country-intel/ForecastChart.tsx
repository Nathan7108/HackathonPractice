"use client";

import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Area, AreaChart, CartesianGrid } from "recharts";
import { Badge } from "@/components/ui/badge";
import type { ForecastData } from "@/lib/types";

type Props = { forecast: ForecastData };

export function ForecastChart({ forecast }: Props) {
  const data = [
    { period: "30d", score: forecast.score30d, low: forecast.score30d - 3, high: forecast.score30d + 3 },
    { period: "60d", score: forecast.score60d, low: forecast.score60d - 3, high: forecast.score60d + 3 },
    { period: "90d", score: forecast.score90d, low: forecast.score90d - 3, high: forecast.score90d + 3 },
  ];
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">90-Day Forecast</h2>
      <div className="h-[160px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="period" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Area type="monotone" dataKey="high" stackId="1" stroke="none" fill="hsl(var(--primary) / 0.2)" />
            <Area type="monotone" dataKey="low" stackId="1" stroke="none" fill="hsl(var(--background))" />
            <Line type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <Badge variant="outline">{forecast.trend}</Badge>
    </section>
  );
}
