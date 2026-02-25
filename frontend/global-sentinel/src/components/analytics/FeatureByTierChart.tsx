"use client";

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Legend, CartesianGrid } from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { FEATURE_BY_TIER } from "@/lib/analytics-data";

export function FeatureByTierChart() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Feature Importance by Tier</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={FEATURE_BY_TIER}
              margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="feature" width={100} tick={{ fontSize: 10 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="value" name="Importance" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
