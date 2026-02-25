"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { PanelHeader } from "@/components/dashboard/PanelHeader";
import { SENTIMENT_TREND_30D } from "@/lib/dashboard-data";

export function SentimentTrendChart() {
  return (
    <Card className="p-0 border border-border rounded-sm shadow-none h-full min-h-0 flex flex-col">
      <PanelHeader title="Global Headline Sentiment (30d)" />
      <CardContent className="p-3 flex-1 min-h-0">
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={SENTIMENT_TREND_30D}
              margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
              stackOffset="expand"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="day"
                tick={{ fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis hide domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area
                type="monotone"
                dataKey="escalatory"
                name="Escalatory"
                stackId="1"
                stroke="#dc2626"
                fill="#dc2626"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="neutral"
                name="Neutral"
                stackId="1"
                stroke="#94a3b8"
                fill="#94a3b8"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="deescalatory"
                name="De-escalatory"
                stackId="1"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
