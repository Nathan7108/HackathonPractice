"use client";

import { KpiCard } from "./KpiCard";
import { KPI_SPARKLINE_DATA } from "@/lib/dashboard-data";

const KPIS = [
  {
    value: 47,
    label: "Global Threat Index",
    trend: { direction: "up" as const, delta: "3" },
    sparklineData: KPI_SPARKLINE_DATA[0],
  },
  {
    value: "12",
    label: "Active Anomalies",
    sparklineData: KPI_SPARKLINE_DATA[1],
  },
  {
    value: "17",
    label: "HIGH+ Risk Countries",
    trend: { direction: "up" as const, delta: "2" },
    sparklineData: KPI_SPARKLINE_DATA[2],
  },
  {
    value: "4",
    label: "Escalation Alerts (24h)",
    sparklineData: KPI_SPARKLINE_DATA[3],
  },
  {
    value: "98%",
    label: "Model Health",
    sparklineData: KPI_SPARKLINE_DATA[4],
  },
];

export function KpiCardRow() {
  return (
    <div className="flex flex-wrap gap-4">
      {KPIS.map((kpi) => (
        <KpiCard
          key={kpi.label}
          value={kpi.value}
          label={kpi.label}
          trend={kpi.trend}
          sparklineData={kpi.sparklineData}
        />
      ))}
    </div>
  );
}
