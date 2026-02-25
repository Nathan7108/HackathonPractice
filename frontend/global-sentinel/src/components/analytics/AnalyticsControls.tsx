"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";

const REGIONS = ["Middle East", "Europe", "Asia", "Africa", "Americas", "Asia Pacific"];
const RISK_TIERS = ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"];
const TIME_RANGES = ["7d", "30d", "90d", "1y", "All"];

export function AnalyticsControls() {
  return (
    <div className="flex flex-wrap items-center gap-4 p-4 border-b border-border bg-card rounded-lg">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground shrink-0">Region</span>
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm" aria-label="Region filter">
          <option>All regions</option>
          {REGIONS.map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground shrink-0">Risk tier</span>
        <div className="flex flex-wrap gap-3">
          {RISK_TIERS.map((tier) => (
            <label key={tier} className="flex items-center gap-1.5 text-sm cursor-pointer">
              <Checkbox aria-label={tier} />
              <span>{tier}</span>
            </label>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground shrink-0">Time range</span>
        <div className="flex gap-1">
          {TIME_RANGES.map((tr) => (
            <button
              key={tr}
              type="button"
              className={cn(
                "px-2 py-1 rounded text-xs font-medium",
                tr === "30d" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
              )}
            >
              {tr}
            </button>
          ))}
        </div>
      </div>
      <Button variant="outline" size="sm">Compare</Button>
      <Button variant="outline" size="sm">Export CSV</Button>
    </div>
  );
}
