"use client";

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
import { ALL_COUNTRIES_TABLE } from "@/lib/analytics-data";
import { cn } from "@/lib/utils";

export function CountryRiskTable() {
  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold">Country Risk Table</h3>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Flag</TableHead>
              <TableHead>Country</TableHead>
              <TableHead>Region</TableHead>
              <TableHead className="text-right">Risk Score</TableHead>
              <TableHead>Level</TableHead>
              <TableHead className="text-right">Confidence</TableHead>
              <TableHead className="w-8">Anom</TableHead>
              <TableHead className="text-right">7d</TableHead>
              <TableHead className="text-right">30d</TableHead>
              <TableHead className="text-right">Battles</TableHead>
              <TableHead className="text-right">Fatalities</TableHead>
              <TableHead>Sentiment</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ALL_COUNTRIES_TABLE.map((row) => (
              <TableRow key={row.country}>
                <TableCell>{row.flag}</TableCell>
                <TableCell className="font-medium">{row.country}</TableCell>
                <TableCell className="text-muted-foreground">{row.region}</TableCell>
                <TableCell className="text-right font-mono">{row.riskScore}</TableCell>
                <TableCell>
                  <Badge variant={row.riskLevel === "HIGH" || row.riskLevel === "CRITICAL" ? "destructive" : "secondary"} className="text-xs">
                    {row.riskLevel}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">{(row.confidence * 100).toFixed(0)}%</TableCell>
                <TableCell>
                  {row.anomaly ? (
                    <span className="h-2 w-2 rounded-full bg-amber-500 block" aria-hidden />
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </TableCell>
                <TableCell className={cn("text-right font-mono", row.change7d > 0 ? "text-red-600" : row.change7d < 0 ? "text-green-600" : "")}>
                  {row.change7d > 0 ? "+" : ""}{row.change7d}
                </TableCell>
                <TableCell className={cn("text-right font-mono", row.change30d > 0 ? "text-red-600" : row.change30d < 0 ? "text-green-600" : "")}>
                  {row.change30d > 0 ? "+" : ""}{row.change30d}
                </TableCell>
                <TableCell className="text-right font-mono">{row.battles}</TableCell>
                <TableCell className="text-right font-mono">{row.fatalities}</TableCell>
                <TableCell className="text-muted-foreground capitalize">{row.sentiment}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
