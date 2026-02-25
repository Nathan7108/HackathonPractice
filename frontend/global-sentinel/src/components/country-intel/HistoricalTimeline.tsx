"use client";

import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from "recharts";

const HISTORICAL = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  score: 75 + Math.sin(i * 0.4) * 15 + i * 0.5,
}));

export function HistoricalTimeline() {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Historical Risk Timeline</h2>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={HISTORICAL} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Line type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
