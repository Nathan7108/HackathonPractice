"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, ChevronLeft, ChevronRight, AlertTriangle, List } from "lucide-react";
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

export function Sidebar({
  isCollapsed: isCollapsedProp,
  setIsCollapsed: setIsCollapsedProp,
  skipToggle,
}: SidebarProps = {}) {
  const pathname = usePathname();
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isControlled = skipToggle && typeof isCollapsedProp === "boolean";
  const isCollapsed = isControlled ? isCollapsedProp! : internalCollapsed;
  const setIsCollapsed = isControlled ? setIsCollapsedProp ?? (() => {}) : setInternalCollapsed;

  const content = (
    <div className="flex flex-1 min-h-0 flex-col overflow-hidden min-w-[260px] whitespace-nowrap">
      {/* Header: search row + toggle (absolute). Fixed height so no vertical shift when input hidden. */}
      <div className="relative shrink-0 border-b border-slate-200/80 min-h-[60px]">
        <div className="flex items-center absolute inset-0 py-3">
          <div className="w-[52px] shrink-0 flex items-center justify-center">
            <Search className="h-4 w-4 text-slate-500" aria-hidden />
          </div>
          {!isCollapsed && (
            <div className="min-w-0 flex-1 pr-12">
              <div className="relative w-full">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 pointer-events-none" />
                <Input
                  type="search"
                  placeholder="Search countries..."
                  className="pl-8 h-9 w-full bg-white border-slate-200 focus-visible:ring-slate-400"
                  aria-label="Search countries"
                />
              </div>
            </div>
          )}
        </div>
        {!skipToggle && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute top-2 right-2 h-8 w-8 shrink-0 text-slate-600 hover:bg-slate-200/60 hover:text-slate-900 z-10"
            onClick={() => setIsCollapsed((c: boolean) => !c)}
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        )}
      </div>

      {/* Scroll: watchlist + anomaly + stats — same JSX always, overflow clips when narrow */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
        <h2 className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2 px-3 pt-3 min-h-[1.25rem]">
          <List className="h-4 w-4 shrink-0 text-slate-500" aria-hidden />
          <span className={isCollapsed ? "invisible" : ""}>Watchlist</span>
        </h2>
        <ul className="space-y-0.5 px-3 pb-2">
          {WATCHLIST_COUNTRIES.map((country) => {
            const isActive = pathname === `/country/${country.code}`;
            return (
              <li key={country.code}>
                <Link
                  href={`/country/${country.code}`}
                  className="flex items-center gap-2 py-1.5 text-sm transition-colors text-slate-700 hover:bg-slate-200/60"
                >
                  <span
                    className={cn(
                      "flex items-center gap-2 rounded-md py-1 pr-1.5 pl-1.5 -ml-0.5 shrink-0",
                      isActive && "bg-slate-200/80 text-slate-900 font-medium"
                    )}
                  >
                    <span
                      className={cn("h-2 w-2 rounded-full shrink-0", RISK_DOT_COLORS[country.riskLevel])}
                      aria-hidden
                    />
                    <span className={cn("shrink-0 text-sm", isCollapsed ? "" : "invisible w-0 overflow-hidden")}>
                      {country.code}
                    </span>
                  </span>
                  <span className={cn("truncate min-w-0 text-sm", isCollapsed ? "invisible" : "")}>
                    {country.flag} {country.name}
                  </span>
                  <span className={cn("ml-auto text-slate-500 shrink-0 text-sm", isCollapsed ? "invisible" : "")}>
                    {country.riskScore}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>

        <h2 className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2 px-3 pt-4 min-h-[1.25rem]">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" aria-hidden />
          <span className={isCollapsed ? "invisible" : ""}>Anomaly Alerts</span>
        </h2>
        <ul className="space-y-2 px-3 pb-4">
          {ANOMALY_ALERTS.map((alert, i) => (
            <li
              key={i}
              className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm bg-slate-100 border border-slate-200/60"
            >
              <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
              <span className="truncate font-medium">{alert.country}</span>
              <span className="text-xs text-slate-500 shrink-0">{alert.severity}</span>
              <span className="ml-auto text-xs text-slate-500 shrink-0">{alert.time}</span>
            </li>
          ))}
        </ul>

        <h2 className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-2 px-3 pt-2 min-h-[1.25rem]">
          <span className={isCollapsed ? "invisible" : ""}>Global Stats</span>
        </h2>
        <div className="grid grid-cols-2 gap-2 px-3 pb-3">
          {GLOBAL_STATS.map((stat) => (
            <div key={stat.label} className="rounded-md border border-slate-200/80 bg-white/80 p-2 text-center min-h-[52px] flex flex-col justify-center">
              <div className="text-sm font-semibold text-slate-800">{stat.value}</div>
              <div className={cn("text-[10px] text-slate-500", isCollapsed && "invisible")}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  if (skipToggle) {
    return (
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden min-w-0">
        {content}
      </div>
    );
  }

  return (
    <aside
      className={cn(
        "flex-shrink-0 sticky top-[52px] h-[calc(100vh-52px)] border-r border-slate-200/80 bg-[#F6F9FB] flex flex-col overflow-hidden transition-all duration-200 whitespace-nowrap",
        isCollapsed ? "w-[60px]" : "w-[260px]"
      )}
      aria-label="App sidebar"
      aria-expanded={!isCollapsed}
    >
      {content}
    </aside>
  );
}
