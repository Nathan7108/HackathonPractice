"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, FileDown } from "lucide-react";
import type { ReportEntry } from "@/lib/reports-data";

type Props = { report: ReportEntry };

export function ReportCard({ report }: Props) {
  const [expanded, setExpanded] = useState(false);
  return (
    <Card>
      <CardHeader className="py-3 px-4 flex flex-row items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xl shrink-0">{report.flag}</span>
          <div>
            <p className="font-medium">{report.country}</p>
            <p className="text-xs text-muted-foreground">{report.generatedAt}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-lg font-bold tabular-nums">{report.riskScore}</span>
          <Badge variant={report.riskLevel === "HIGH" || report.riskLevel === "CRITICAL" ? "destructive" : "secondary"}>
            {report.riskLevel}
          </Badge>
          <Badge variant="outline">{report.status}</Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 px-4 pb-3">
        <p className="text-sm text-muted-foreground">
          {report.briefPreview[0]}
          <br />
          {report.briefPreview[1]}
        </p>
        <div className="mt-3 flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-1">
            <FileDown className="h-3 w-3" />
            Export as PDF
          </Button>
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {expanded ? "Collapse" : "Expand"}
          </button>
        </div>
        {expanded && (
          <div className="mt-3 pt-3 border-t border-border text-sm">
            {report.fullBrief}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
