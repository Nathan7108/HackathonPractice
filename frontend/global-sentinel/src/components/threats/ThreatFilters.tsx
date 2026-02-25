"use client";

import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { cn } from "@/lib/utils";

const REGIONS = ["All", "Middle East", "Europe", "Asia", "Africa", "Americas"];
const SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
const TYPES = ["All", "Anomaly", "Escalation", "De-escalation", "New Intel"];

export function ThreatFilters() {
  return (
    <div className="flex flex-wrap items-center gap-4 p-4 border-b border-border bg-card rounded-lg">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground shrink-0">Region</span>
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm" aria-label="Region">
          {REGIONS.map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground shrink-0">Severity</span>
        <div className="flex gap-2">
          {SEVERITIES.map((s) => (
            <label key={s} className="flex items-center gap-1 text-xs cursor-pointer">
              <input type="checkbox" aria-label={s} />
              {s}
            </label>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground shrink-0">Type</span>
        <div className="flex gap-1">
          {TYPES.map((t) => (
            <button key={t} type="button" className={cn("px-2 py-1 rounded text-xs", t === "All" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80")}>
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input type="search" placeholder="Search events..." className="pl-8" aria-label="Search" />
      </div>
    </div>
  );
}
