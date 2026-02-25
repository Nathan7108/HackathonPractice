"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { TOP_ESCALATING, TOP_DEESCALATING } from "@/lib/dashboard-data";

export function EscalationLists() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold">Top 5 Escalating</h3>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {TOP_ESCALATING.map((item) => (
              <li
                key={item.country}
                className="flex items-center justify-between text-sm"
              >
                <span>{item.country}</span>
                <span className="font-medium text-red-600">+{item.delta}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold">Top 5 De-escalating</h3>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {TOP_DEESCALATING.map((item) => (
              <li
                key={item.country}
                className="flex items-center justify-between text-sm"
              >
                <span>{item.country}</span>
                <span className="font-medium text-green-600">{item.delta}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
