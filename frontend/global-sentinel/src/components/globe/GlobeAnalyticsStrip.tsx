"use client";

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, LineChart, Line } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { RISK_DISTRIBUTION } from "@/lib/dashboard-data";
import { TOP_ESCALATING } from "@/lib/dashboard-data";
import { SENTIMENT_TREND_30D } from "@/lib/dashboard-data";

const ANOMALY_TREND = [
  { day: "M", count: 2 },
  { day: "T", count: 3 },
  { day: "W", count: 1 },
  { day: "T", count: 4 },
  { day: "F", count: 2 },
  { day: "S", count: 3 },
  { day: "S", count: 2 },
];

export function GlobeAnalyticsStrip() {
  return (
    <div className="absolute bottom-4 left-4 right-4 z-10 flex gap-4 overflow-x-auto pb-2">
      <Card className="min-w-[200px] h-[140px] shrink-0">
        <CardHeader className="py-2 px-3">
          <h3 className="text-xs font-semibold">Risk Distribution</h3>
        </CardHeader>
        <CardContent className="px-3 pt-0 h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={RISK_DISTRIBUTION} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <Bar dataKey="count" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]} />
              <XAxis dataKey="tier" tick={{ fontSize: 8 }} hide />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card className="min-w-[200px] h-[140px] shrink-0">
        <CardHeader className="py-2 px-3">
          <h3 className="text-xs font-semibold">Anomaly Trend</h3>
        </CardHeader>
        <CardContent className="px-3 pt-0 h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ANOMALY_TREND} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <Line type="monotone" dataKey="count" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={{ r: 2 }} />
              <XAxis dataKey="day" tick={{ fontSize: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card className="min-w-[200px] h-[140px] shrink-0">
        <CardHeader className="py-2 px-3">
          <h3 className="text-xs font-semibold">Top Escalating</h3>
        </CardHeader>
        <CardContent className="px-3 pt-0">
          <ul className="text-xs space-y-0.5">
            {TOP_ESCALATING.slice(0, 4).map((item) => (
              <li key={item.country} className="flex justify-between">
                <span>{item.country}</span>
                <span className="text-red-600 font-medium">+{item.delta}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
      <Card className="min-w-[200px] h-[140px] shrink-0">
        <CardHeader className="py-2 px-3">
          <h3 className="text-xs font-semibold">Headline Volume</h3>
        </CardHeader>
        <CardContent className="px-3 pt-0 h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={SENTIMENT_TREND_30D.slice(-7)} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <Bar dataKey="escalatory" stackId="a" fill="hsl(0, 84%, 60%)" radius={[2, 2, 0, 0]} />
              <Bar dataKey="neutral" stackId="a" fill="hsl(var(--muted))" radius={[0, 0, 0, 0]} />
              <Bar dataKey="deescalatory" stackId="a" fill="hsl(142, 76%, 36%)" radius={[0, 0, 2, 2]} />
              <XAxis dataKey="day" tick={{ fontSize: 8 }} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
