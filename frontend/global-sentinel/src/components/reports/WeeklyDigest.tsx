"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";

const TOP_CHANGES = [
  { country: "Sudan", change: "+14", direction: "up" },
  { country: "Myanmar", change: "+11", direction: "up" },
  { country: "Colombia", change: "-8", direction: "down" },
  { country: "Ethiopia", change: "-6", direction: "down" },
  { country: "Iran", change: "+6", direction: "up" },
];

export function WeeklyDigest() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Weekly Digest</h3>
        <p className="text-xs text-muted-foreground">Top 5 risk changes (PLACEHOLDER)</p>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {TOP_CHANGES.map((item) => (
            <li key={item.country} className="flex justify-between text-sm">
              <span>{item.country}</span>
              <span className={item.direction === "up" ? "text-red-600 font-medium" : "text-green-600 font-medium"}>
                {item.change}
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
