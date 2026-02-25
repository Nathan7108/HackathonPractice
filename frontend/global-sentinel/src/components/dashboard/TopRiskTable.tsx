"use client";

import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { PanelHeader } from "@/components/dashboard/PanelHeader";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { KPI_SPARKLINE_DATA } from "@/lib/dashboard-data";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

const RISK_BADGE_CLASS: Record<RiskLevel, string> = {
  LOW: "bg-green-100 text-green-800",
  MODERATE: "bg-yellow-100 text-yellow-800",
  ELEVATED: "bg-orange-100 text-orange-800",
  HIGH: "bg-red-100 text-red-800",
  CRITICAL: "bg-red-200 text-red-900",
};

function SparklineSvg({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const w = 48;
  const h = 20;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 2;
  const points = data
    .map((v, i) => {
      const x = padding + (i / (data.length - 1)) * (w - 2 * padding);
      const y = h - padding - ((v - min) / range) * (h - 2 * padding);
      return `${x},${y}`;
    })
    .join(" ");
  const positiveTrend = data[data.length - 1] >= data[0];
  const strokeColor = positiveTrend ? "#22c55e" : "#d63031";
  return (
    <svg width={w} height={h} className="shrink-0" aria-hidden>
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

const TOP_10 = [
  ...WATCHLIST_COUNTRIES.slice(0, 8),
  { code: "PL1", name: "PLACEHOLDER Country A", flag: "🏳", riskScore: 9, riskLevel: "LOW" as RiskLevel },
  { code: "PL2", name: "PLACEHOLDER Country B", flag: "🏳", riskScore: 5, riskLevel: "LOW" as RiskLevel },
].slice(0, 10);

export function TopRiskTable() {
  return (
    <Card className="p-0 border border-border rounded-sm shadow-none h-full min-h-0 flex flex-col">
      <PanelHeader title="Top 10 Highest Risk" />
      <CardContent className="p-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">#</TableHead>
              <TableHead>Country</TableHead>
              <TableHead className="text-right">Score</TableHead>
              <TableHead>Level</TableHead>
              <TableHead className="w-20">Trend</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {TOP_10.map((c, i) => (
              <TableRow key={c.code}>
                <TableCell className="font-mono text-muted-foreground py-1.5">{i + 1}</TableCell>
                <TableCell className="py-1.5">
                  {c.code.startsWith("PL") ? (
                    <span>{c.flag} {c.name}</span>
                  ) : (
                    <Link
                      href={`/country/${c.code}`}
                      className="hover:underline font-medium"
                    >
                      {c.flag} {c.name}
                    </Link>
                  )}
                </TableCell>
                <TableCell className="text-right font-mono py-1.5">{c.riskScore}</TableCell>
                <TableCell className="py-1.5">
                  <Badge variant="secondary" className={cn("text-xs", RISK_BADGE_CLASS[c.riskLevel])}>
                    {c.riskLevel}
                  </Badge>
                </TableCell>
                <TableCell className="py-1.5">
                  <SparklineSvg data={KPI_SPARKLINE_DATA[i % KPI_SPARKLINE_DATA.length]} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
