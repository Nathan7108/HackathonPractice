"use client";

import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Legend, CartesianGrid } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { RISK_TRENDS } from "@/lib/analytics-data";

const COLORS = ["hsl(var(--primary))", "hsl(var(--chart-2))", "hsl(var(--chart-3))", "hsl(var(--chart-4))", "hsl(var(--chart-5))"];

export function RiskTrendLines() {
  const keys = Object.keys(RISK_TRENDS).slice(0, 5);
  const data = (RISK_TRENDS.ukraine ?? []).map((_, i) => {
    const point: Record<string, string | number> = { week: `W${i + 1}` };
    keys.forEach((k, j) => {
      point[k] = RISK_TRENDS[k]?.[i]?.score ?? 0;
    });
    return point;
  });
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Risk Trend Lines</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="week" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {keys.map((k, i) => (
                <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
