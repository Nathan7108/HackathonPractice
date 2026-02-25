"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SCATTER_DATA } from "@/lib/analytics-data";

export function FeatureScatterPlot() {
  const data = SCATTER_DATA.map((d) => ({
    x: d.gdpGrowth,
    y: d.riskScore,
    z: 400,
    name: d.country,
  }));
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Feature Correlation (GDP Growth vs Risk)</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" dataKey="x" name="GDP Growth" tick={{ fontSize: 10 }} />
              <YAxis type="number" dataKey="y" name="Risk Score" tick={{ fontSize: 10 }} />
              <ZAxis type="number" dataKey="z" range={[100, 400]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={data} fill="hsl(var(--primary))" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
