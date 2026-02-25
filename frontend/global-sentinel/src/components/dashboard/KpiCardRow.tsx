"use client";

import { useState, useEffect } from "react";
import { KpiCard } from "./KpiCard";
import { KPI_SPARKLINE_DATA } from "@/lib/dashboard-data";
import { fetchDashboardSummary } from "@/lib/api";

/** Placeholder config when API data is null (loading or error). */
const PLACEHOLDER_CONFIG = [
  {
    value: 47,
    label: "Global Threat Index",
    trend: { direction: "up" as const, delta: "3" },
    sparklineData: KPI_SPARKLINE_DATA[0],
    bgClass: "bg-amber-100",
    borderLeftClass: "border-l-amber-500",
    iconVariant: "alert" as const,
    iconClassName: "text-amber-500",
    sparklineColor: "#f59e0b",
  },
  {
    value: "12",
    label: "Active Anomalies",
    sparklineData: KPI_SPARKLINE_DATA[1],
    bgClass: "bg-red-50",
    borderLeftClass: "border-l-red-500",
    iconVariant: "alert" as const,
    iconClassName: "text-red-500",
    sparklineColor: "#ef4444",
  },
  {
    value: "17",
    label: "HIGH+ Risk Countries",
    trend: { direction: "up" as const, delta: "2" },
    sparklineData: KPI_SPARKLINE_DATA[2],
    bgClass: "bg-red-50",
    borderLeftClass: "border-l-red-500",
    iconVariant: "alert" as const,
    iconClassName: "text-red-500",
    sparklineColor: "#ef4444",
  },
  {
    value: "4",
    label: "Escalation Alerts (24h)",
    sparklineData: KPI_SPARKLINE_DATA[3],
    bgClass: "bg-orange-50",
    borderLeftClass: "border-l-orange-500",
    iconVariant: "alert" as const,
    iconClassName: "text-orange-500",
    sparklineColor: "#f97316",
  },
  {
    value: "98%",
    label: "Model Health",
    sparklineData: KPI_SPARKLINE_DATA[4],
    bgClass: "bg-green-50",
    borderLeftClass: "border-l-green-600",
    iconVariant: "healthy" as const,
    iconClassName: "text-green-600",
    sparklineColor: "#16a34a",
  },
];

type DashboardSummary = {
  globalThreatIndex: number;
  globalThreatIndexDelta: number;
  activeAnomalies: number;
  highPlusCountries: number;
  highPlusCountriesDelta: number;
  escalationAlerts24h: number;
  modelHealth: number;
  countries?: unknown[];
};

function trendFromDelta(delta: number): { direction: "up" | "down"; delta: string } | undefined {
  if (delta === 0) return undefined;
  return {
    direction: delta > 0 ? "up" : "down",
    delta: String(Math.abs(delta)),
  };
}

export function KpiCardRow() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardSummary()
      .then((json) => {
        setData(json);
      })
      .catch((err) => {
        console.error("Dashboard summary fetch failed:", err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const usePlaceholder = data === null;
  const config = usePlaceholder
    ? PLACEHOLDER_CONFIG
    : [
        {
          value: data!.globalThreatIndex,
          label: "Global Threat Index",
          trend: trendFromDelta(data!.globalThreatIndexDelta),
          sparklineData: KPI_SPARKLINE_DATA[0],
          bgClass: "bg-amber-100",
          borderLeftClass: "border-l-amber-500",
          iconVariant: "alert" as const,
          iconClassName: "text-amber-500",
          sparklineColor: "#f59e0b",
        },
        {
          value: String(data!.activeAnomalies),
          label: "Active Anomalies",
          sparklineData: KPI_SPARKLINE_DATA[1],
          bgClass: "bg-red-50",
          borderLeftClass: "border-l-red-500",
          iconVariant: "alert" as const,
          iconClassName: "text-red-500",
          sparklineColor: "#ef4444",
        },
        {
          value: String(data!.highPlusCountries),
          label: "HIGH+ Risk Countries",
          trend: trendFromDelta(data!.highPlusCountriesDelta),
          sparklineData: KPI_SPARKLINE_DATA[2],
          bgClass: "bg-red-50",
          borderLeftClass: "border-l-red-500",
          iconVariant: "alert" as const,
          iconClassName: "text-red-500",
          sparklineColor: "#ef4444",
        },
        {
          value: String(data!.escalationAlerts24h),
          label: "Escalation Alerts (24h)",
          sparklineData: KPI_SPARKLINE_DATA[3],
          bgClass: "bg-orange-50",
          borderLeftClass: "border-l-orange-500",
          iconVariant: "alert" as const,
          iconClassName: "text-orange-500",
          sparklineColor: "#f97316",
        },
        {
          value: `${data!.modelHealth}%`,
          label: "Model Health",
          sparklineData: KPI_SPARKLINE_DATA[4],
          bgClass: "bg-green-50",
          borderLeftClass: "border-l-green-600",
          iconVariant: "healthy" as const,
          iconClassName: "text-green-600",
          sparklineColor: "#16a34a",
        },
      ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-0.5">
      {config.map((kpi) => (
        <KpiCard
          key={kpi.label}
          value={kpi.value}
          label={kpi.label}
          trend={kpi.trend}
          sparklineData={kpi.sparklineData}
          bgClass={kpi.bgClass}
          borderLeftClass={kpi.borderLeftClass}
          iconVariant={kpi.iconVariant}
          iconClassName={kpi.iconClassName}
          sparklineColor={kpi.sparklineColor}
          isLoading={loading}
        />
      ))}
    </div>
  );
}
