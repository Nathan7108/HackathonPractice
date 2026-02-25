"use client";

import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

const RISK_DOT: Record<RiskLevel, string> = {
  LOW: "bg-green-500",
  MODERATE: "bg-yellow-500",
  ELEVATED: "bg-orange-500",
  HIGH: "bg-red-500",
  CRITICAL: "bg-red-700",
};

type Props = { onCountrySelect: (code: string) => void; selectedCode?: string | null };

export function GlobeOverlay({ onCountrySelect, selectedCode }: Props) {
  return (
    <div className="absolute top-4 left-4 z-10 w-[260px] rounded-lg border border-border bg-white/95 backdrop-blur-sm shadow-lg p-3">
      <div className="relative mb-3">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search countries..."
          className="pl-8 h-9"
          aria-label="Search countries"
        />
      </div>
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
        Watchlist
      </h2>
      <ul className="space-y-0.5">
        {WATCHLIST_COUNTRIES.map((c) => (
          <li key={c.code}>
            <button
              type="button"
              onClick={() => onCountrySelect(c.code)}
              className={cn(
                "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-left transition-colors",
                selectedCode === c.code ? "bg-primary/10 text-primary font-medium" : "hover:bg-muted"
              )}
            >
              <span className={cn("h-2 w-2 rounded-full shrink-0", RISK_DOT[c.riskLevel])} />
              <span className="truncate">{c.flag} {c.name}</span>
              <span className="ml-auto text-muted-foreground text-xs">{c.riskScore}</span>
            </button>
          </li>
        ))}
      </ul>
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mt-4 mb-2">
        Anomaly Alerts
      </h2>
      <ul className="space-y-1.5 text-sm">
        <li className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
          <span>Ukraine — HIGH</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
          <span>Iran — HIGH</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
          <span>Pakistan — ELEVATED</span>
        </li>
      </ul>
    </div>
  );
}
