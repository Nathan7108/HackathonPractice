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
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

const RISK_BADGE_VARIANT: Record<RiskLevel, "default" | "secondary" | "destructive" | "outline"> = {
  LOW: "secondary",
  MODERATE: "outline",
  ELEVATED: "default",
  HIGH: "destructive",
  CRITICAL: "destructive",
};

const TOP_10 = [
  ...WATCHLIST_COUNTRIES.slice(0, 8),
  { code: "PL1", name: "PLACEHOLDER Country A", flag: "🏳", riskScore: 9, riskLevel: "LOW" as RiskLevel },
  { code: "PL2", name: "PLACEHOLDER Country B", flag: "🏳", riskScore: 5, riskLevel: "LOW" as RiskLevel },
].slice(0, 10);

export function TopRiskTable() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Top 10 Highest Risk</h3>
      </CardHeader>
      <CardContent className="p-0">
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
                <TableCell className="font-mono text-muted-foreground">{i + 1}</TableCell>
                <TableCell>
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
                <TableCell className="text-right font-mono">{c.riskScore}</TableCell>
                <TableCell>
                  <Badge variant={RISK_BADGE_VARIANT[c.riskLevel]} className="text-xs">
                    {c.riskLevel}
                  </Badge>
                </TableCell>
                <TableCell>
                  <div className="h-6 w-14 bg-muted rounded" title="PLACEHOLDER sparkline" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
