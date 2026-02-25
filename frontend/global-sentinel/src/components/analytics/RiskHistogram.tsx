"use client";

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { RISK_HISTOGRAM } from "@/lib/analytics-data";

export function RiskHistogram() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Risk Score Distribution</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={RISK_HISTOGRAM} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <XAxis dataKey="range" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
