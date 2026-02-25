"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { CALENDAR_HEATMAP } from "@/lib/analytics-data";

export function ConflictHeatmap() {
  const maxVal = Math.max(...CALENDAR_HEATMAP.flatMap((r) => r.weeks));
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Conflict Event Heatmap</h3>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {CALENDAR_HEATMAP.map((row) => (
            <div key={row.country} className="flex gap-1">
              {row.weeks.map((val, i) => (
                <div
                  key={i}
                  className="w-6 h-6 rounded-sm flex items-center justify-center text-[10px]"
                  style={{
                    backgroundColor: `hsl(var(--primary) / ${0.2 + (val / maxVal) * 0.8})`,
                  }}
                  title={`${row.country} W${i + 1}: ${val}`}
                >
                  {val}
                </div>
              ))}
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-2">8 countries × 12 weeks (PLACEHOLDER)</p>
      </CardContent>
    </Card>
  );
}
