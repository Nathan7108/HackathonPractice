"use client";

import { LineChart, Line, ResponsiveContainer } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { CheckCircle, AlertTriangle } from "lucide-react";

type KpiCardProps = {
  value: string | number;
  label: string;
  trend?: { direction: "up" | "down"; delta: string };
  sparklineData?: number[]; 
  /** e.g. bg-amber-100 */
  bgClass: string;
  /** e.g. border-amber-500 */
  borderLeftClass: string;
  /** CheckCircle for healthy, AlertTriangle for warning/alert */
  iconVariant: "healthy" | "alert";
  /** e.g. text-amber-500 */
  iconClassName: string;
  /** Sparkline stroke color (hex or Tailwind color for stroke) */
  sparklineColor: string;
  /** When true, value text gets subtle pulse animation (e.g. while loading). */
  isLoading?: boolean;
};

export function KpiCard({
  value,
  label,
  trend,
  sparklineData = [],
  bgClass,
  borderLeftClass,
  iconVariant,
  iconClassName,
  sparklineColor,
  isLoading = false,
}: KpiCardProps) {
  const data = sparklineData.map((v, i) => ({ x: i, v }));
  const trendUp = trend?.direction === "up";
  const trendDown = trend?.direction === "down";
  const StatusIcon = iconVariant === "healthy" ? CheckCircle : AlertTriangle;

  return (
    <Card
      className={cn("min-w-[180px] flex-1 h-[90px] flex flex-col border border-border rounded-sm shadow-none overflow-hidden p-0 gap-0 border-l-4", bgClass, borderLeftClass)}
    >
      {/* Top row: label + status icon */}
      <div className="flex items-start justify-between py-2 px-3 pb-0">
        <p className="text-[11px] uppercase tracking-wider text-gray-600">{label}</p>
        <StatusIcon className={`h-4 w-4 shrink-0 ${iconClassName}`} aria-hidden />
      </div>

      <CardContent className="py-1 px-3 pt-0 flex-1 flex flex-col min-h-0">
        <div className="flex items-baseline gap-2">
          <span className={cn("text-3xl font-bold tabular-nums text-gray-900", isLoading && "animate-pulse")}>{value}</span>
          {trend && (
            <span
              className={`text-xs font-medium ${
                trendUp ? "text-red-600" : trendDown ? "text-green-600" : "text-gray-500"
              }`}
            >
              {trendUp ? "↑" : trendDown ? "↓" : ""}
              {trend.delta}
            </span>
          )}
        </div>

        {/* Full-width sparkline below number, 30px tall */}
        {data.length > 0 && (
          <div className="w-full h-[30px] mt-1 shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
                <Line
                  type="monotone"
                  dataKey="v"
                  stroke={sparklineColor}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
