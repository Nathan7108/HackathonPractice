"use client";

import { Card, CardContent, CardHeader } from "@/components/ui/card";

const ALERT_COUNTS = [
  { severity: "CRITICAL", count: 2 },
  { severity: "HIGH", count: 8 },
  { severity: "MEDIUM", count: 7 },
  { severity: "LOW", count: 3 },
];

const ACTIVE_COUNTRIES = [
  { country: "Ukraine", count: 12 },
  { country: "Iran", count: 8 },
  { country: "Sudan", count: 6 },
  { country: "Syria", count: 5 },
  { country: "Pakistan", count: 4 },
];

const KEYWORDS = ["PLACEHOLDER", "conflict", "escalation", "ceasefire", "military", "protest", "sanctions", "diplomacy", "intelligence", "risk"];

export function ThreatSidebar() {
  return (
    <div className="space-y-6 w-full max-w-[320px] shrink-0">
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold">Active Alerts Summary</h3>
        </CardHeader>
        <CardContent className="space-y-2">
          {ALERT_COUNTS.map((a) => (
            <div key={a.severity} className="flex justify-between text-sm">
              <span>{a.severity}</span>
              <span className="font-mono">{a.count}</span>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold">Most Active Countries (24h)</h3>
        </CardHeader>
        <CardContent className="space-y-2">
          {ACTIVE_COUNTRIES.map((c) => (
            <div key={c.country} className="flex justify-between text-sm">
              <span>{c.country}</span>
              <span className="font-mono">{c.count}</span>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold">Trending Keywords</h3>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-1.5">
            {KEYWORDS.map((k) => (
              <span key={k} className="text-xs px-2 py-0.5 rounded-full bg-muted">
                {k}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
