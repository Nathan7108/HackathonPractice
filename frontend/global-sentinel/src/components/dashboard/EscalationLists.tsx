"use client";

import { Card, CardContent } from "@/components/ui/card";
import { PanelHeader } from "@/components/dashboard/PanelHeader";
import { TOP_ESCALATING, TOP_DEESCALATING } from "@/lib/dashboard-data";

export function EscalationLists() {
  return (
    <div className="grid grid-cols-2 gap-0.5 h-full min-h-0">
      <Card className="p-0 border border-border rounded-sm shadow-none h-full min-h-0 flex flex-col">
        <PanelHeader title="Top 5 Escalating" />
        <CardContent className="p-3">
          <ul className="space-y-1 text-sm">
            {TOP_ESCALATING.map((item) => (
              <li
                key={item.country}
                className="flex items-center justify-between"
              >
                <span>{item.country}</span>
                <span className="font-medium text-red-600 tabular-nums">+{item.delta}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
      <Card className="p-0 border border-border rounded-sm shadow-none h-full min-h-0 flex flex-col">
        <PanelHeader title="Top 5 De-escalating" />
        <CardContent className="p-3">
          <ul className="space-y-1 text-sm">
            {TOP_DEESCALATING.map((item) => (
              <li
                key={item.country}
                className="flex items-center justify-between"
              >
                <span>{item.country}</span>
                <span className="font-medium text-green-600 tabular-nums">{item.delta}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
