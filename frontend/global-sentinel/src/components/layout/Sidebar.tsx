"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Search, ChevronLeft, ChevronRight, AlertTriangle } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

type SidebarProps = {
  isCollapsed?: boolean;
  setIsCollapsed?: (value: boolean | ((prev: boolean) => boolean)) => void;
  skipToggle?: boolean;
};

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
  { country: "PLACEHOLDER Region A", severity: "MODERATE", time: "3h ago" },
  { country: "PLACEHOLDER Region B", severity: "ELEVATED", time: "6h ago" },
];

const GLOBAL_STATS = [
  { label: "Countries", value: "200+" },
  { label: "Anomalies", value: "12" },
  { label: "Accuracy", value: "98%" },
  { label: "Models", value: "201" },
];

export function Sidebar({ isCollapsed: isCollapsedProp, setIsCollapsed: setIsCollapsedProp, skipToggle }: SidebarProps = {}) {
  const pathname = usePathname();
  const router = useRouter();
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isControlled = skipToggle && typeof isCollapsedProp === "boolean";
  const isCollapsed = isControlled ? isCollapsedProp! : internalCollapsed;
  const setIsCollapsed = isControlled ? (setIsCollapsedProp ?? (() => {})) : setInternalCollapsed;

  const handleRailCountryClick = (code: string) => {
    setIsCollapsed(false);
    router.push(`/country/${code}`);
  };

  const bodyContent = (
    <>
      <div className="p-3 shrink-0">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <Input
                type="search"
                placeholder="Search countries..."
                className="pl-8 h-9 bg-white border-slate-200 focus-visible:ring-slate-400"
                aria-label="Search countries"
              />
            </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-6">
              {/* Watchlist */}
              <section>
                <h2 className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2">
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
                              ? "bg-slate-200/80 text-slate-900 font-medium"
                              : "text-slate-700 hover:bg-slate-200/60"
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
                          <span className="ml-auto text-slate-500 text-xs shrink-0">
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
                <h2 className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2">
                  Anomaly Alerts
                </h2>
                <ul className="space-y-2">
                  {ANOMALY_ALERTS.map((alert, i) => (
                    <li
                      key={i}
                      className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm bg-slate-100 border border-slate-200/60"
                    >
                      <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse shrink-0" />
                      <span className="truncate font-medium">{alert.country}</span>
                <span className="text-xs text-slate-500 shrink-0">{alert.severity}</span>
                <span className="ml-auto text-xs text-slate-500 shrink-0">{alert.time}</span>
                    </li>
                  ))}
                </ul>
              </section>
        </div>

        {/* Global stats */}
        <section className="mt-auto pt-4 border-t border-slate-200/80 shrink-0 p-3">
              <h2 className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2">
                Global Stats
              </h2>
              <div className="grid grid-cols-2 gap-2">
                {GLOBAL_STATS.map((stat) => (
                  <div
                    key={stat.label}
                    className="rounded-md border border-slate-200/80 bg-white/80 p-2 text-center"
                  >
                    <div className="text-sm font-semibold text-slate-800">{stat.value}</div>
                    <div className="text-[10px] text-slate-500">{stat.label}</div>
                  </div>
                ))}
              </div>
        </section>
      </div>
    </>
  );

  if (skipToggle) {
    return (
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {isCollapsed ? (
          /* Icon rail: risk dots + 2-letter codes, anomaly badge, compact stats */
          <>
            <div className="flex-1 min-h-0 overflow-y-auto py-2 flex flex-col items-center gap-1">
              {WATCHLIST_COUNTRIES.map((country) => (
                <button
                  key={country.code}
                  type="button"
                  onClick={() => handleRailCountryClick(country.code)}
                  className={cn(
                    "w-full flex items-center justify-center gap-1.5 py-1.5 rounded-md hover:bg-slate-200/60 transition-colors",
                    pathname === `/country/${country.code}` ? "bg-slate-200/80" : ""
                  )}
                  title={country.name}
                >
                  <span
                    className={cn("h-2 w-2 rounded-full shrink-0", RISK_DOT_COLORS[country.riskLevel])}
                    aria-hidden
                  />
                  <span className="text-[11px] font-mono font-medium text-slate-700">{country.code}</span>
                </button>
              ))}
            </div>
            <div className="shrink-0 border-t border-slate-200/80 my-1" />
            <div className="shrink-0 flex items-center justify-center gap-1 px-1 py-1.5 text-amber-600" title="Anomaly alerts">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span className="text-xs font-semibold tabular-nums">{ANOMALY_ALERTS.length}</span>
            </div>
            <div className="shrink-0 flex flex-col items-center gap-0.5 px-1 pb-2 text-[10px] font-mono text-slate-500">
              {GLOBAL_STATS.map((stat) => (
                <span key={stat.label} title={stat.label}>{stat.value}</span>
              ))}
            </div>
          </>
        ) : (
          bodyContent
        )}
      </div>
    );
  }

  return (
    <aside
      className={cn(
        "flex-shrink-0 sticky top-[52px] h-[calc(100vh-52px)] border-r border-slate-200/80 bg-[#F6F9FB] flex flex-col overflow-hidden transition-all duration-200",
        isCollapsed ? "w-12" : "w-[260px]"
      )}
      aria-label="App sidebar"
      aria-expanded={!isCollapsed}
    >
      <div className="p-1 shrink-0 flex items-center justify-center border-b border-slate-200/80">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0 text-slate-600 hover:bg-slate-200/60 hover:text-slate-900"
          onClick={() => setIsCollapsed((c: boolean) => !c)}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>
      {!isCollapsed && bodyContent}
    </aside>
  );
}
