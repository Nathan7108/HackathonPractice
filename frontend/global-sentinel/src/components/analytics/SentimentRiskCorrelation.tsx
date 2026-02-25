"use client";

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SENTIMENT_VS_RISK } from "@/lib/analytics-data";

export function SentimentRiskCorrelation() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Sentiment vs Risk</h3>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={SENTIMENT_VS_RISK} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="week" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line yAxisId="left" type="monotone" dataKey="sentiment" name="Sentiment" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="riskScore" name="Risk Score" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
