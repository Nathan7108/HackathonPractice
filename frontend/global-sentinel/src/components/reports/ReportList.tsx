"use client";

import { ReportCard } from "./ReportCard";
import type { ReportEntry } from "@/lib/reports-data";

type Props = { reports: ReportEntry[] };

export function ReportList({ reports }: Props) {
  return (
    <div className="space-y-3">
      {reports.map((report) => (
        <ReportCard key={report.id} report={report} />
      ))}
    </div>
  );
}
