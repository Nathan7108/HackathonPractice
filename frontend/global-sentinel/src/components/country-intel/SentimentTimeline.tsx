"use client";

import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from "recharts";
import type { HeadlineSentiment } from "@/lib/types";

const SENTIMENT_HISTORY = Array.from({ length: 12 }, (_, i) => ({
  week: `W${i + 1}`,
  escalatory: 25 + Math.sin(i * 0.5) * 10,
  neutral: 50 + Math.cos(i * 0.3) * 8,
  deescalatory: 25 - Math.sin(i * 0.4) * 8,
}));

type Props = { headlines: HeadlineSentiment[] };

export function SentimentTimeline({ headlines }: Props) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Headline Sentiment Timeline</h2>
      <div className="h-[160px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={SENTIMENT_HISTORY} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} stackOffset="expand">
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="week" tick={{ fontSize: 10 }} />
            <YAxis hide domain={[0, 100]} />
            <Area type="monotone" dataKey="escalatory" stackId="1" fill="hsl(0, 84%, 60%)" fillOpacity={0.7} />
            <Area type="monotone" dataKey="neutral" stackId="1" fill="hsl(var(--muted))" fillOpacity={0.7} />
            <Area type="monotone" dataKey="deescalatory" stackId="1" fill="hsl(142, 76%, 36%)" fillOpacity={0.7} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <ul className="space-y-1">
        {headlines.slice(0, 3).map((h, i) => (
          <li key={i} className="flex items-center gap-2 text-xs">
            <span className={`h-1.5 w-1.5 rounded-full ${h.sentiment === "negative" ? "bg-red-500" : h.sentiment === "positive" ? "bg-green-500" : "bg-muted-foreground"}`} />
            {h.text}
          </li>
        ))}
      </ul>
    </section>
  );
}
