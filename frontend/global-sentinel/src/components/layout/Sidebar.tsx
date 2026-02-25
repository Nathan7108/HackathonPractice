"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

const RISK_DOT_COLORS: Record<RiskLevel, string> = {
  LOW: "bg-green-500",
  MODERATE: "bg-yellow-500",
  ELEVATED: "bg-orange-500",
  HIGH: "bg-red-500",
  CRITICAL: "bg-red-700",
};

const ANOMALY_ALERTS = [
  { country: "Ukraine", severity: "HIGH", time: "2h ago" },
  { country: "Iran", severity: "HIGH", time: "5h ago" },
  { country: "Pakistan", severity: "ELEVATED", time: "1d ago" },
];

const GLOBAL_STATS = [
  { label: "Countries", value: "200+" },
  { label: "Anomalies", value: "12" },
  { label: "Accuracy", value: "98%" },
  { label: "Models", value: "201" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-[260px] flex-shrink-0 border-r border-border bg-white flex flex-col min-h-0">
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search countries..."
            className="pl-8 h-9"
            aria-label="Search countries"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-6">
        {/* Watchlist */}
        <section>
          <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
            Watchlist
          </h2>
          <ul className="space-y-0.5">
            {WATCHLIST_COUNTRIES.map((country) => {
              const isActive = pathname === `/country/${country.code}`;
              return (
                <li key={country.code}>
                  <Link
                    href={`/country/${country.code}`}
                    className={cn(
                      "flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-foreground hover:bg-muted"
                    )}
                  >
                    <span
                      className={cn(
                        "h-2 w-2 rounded-full shrink-0",
                        RISK_DOT_COLORS[country.riskLevel]
                      )}
                      aria-hidden
                    />
                    <span className="truncate">{country.flag} {country.name}</span>
                    <span className="ml-auto text-muted-foreground text-xs shrink-0">
                      {country.riskScore}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>

        {/* Anomaly alerts */}
        <section>
          <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
            Anomaly Alerts
          </h2>
          <ul className="space-y-2">
            {ANOMALY_ALERTS.map((alert, i) => (
              <li
                key={i}
                className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm bg-muted/50"
              >
                <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse shrink-0" />
                <span className="truncate font-medium">{alert.country}</span>
                <span className="text-xs text-muted-foreground shrink-0">{alert.severity}</span>
                <span className="ml-auto text-xs text-muted-foreground shrink-0">{alert.time}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* Global stats */}
        <section>
          <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
            Global Stats
          </h2>
          <div className="grid grid-cols-2 gap-2">
            {GLOBAL_STATS.map((stat) => (
              <div
                key={stat.label}
                className="rounded-md border border-border bg-muted/30 p-2 text-center"
              >
                <div className="text-sm font-semibold text-foreground">{stat.value}</div>
                <div className="text-[10px] text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </aside>
  );
}
