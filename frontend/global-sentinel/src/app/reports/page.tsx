import { SidebarLayout } from "@/components/layout/SidebarLayout";
import { ReportControls } from "@/components/reports/ReportControls";
import { WeeklyDigest } from "@/components/reports/WeeklyDigest";
import { ReportList } from "@/components/reports/ReportList";
import { REPORTS } from "@/lib/reports-data";

export default function ReportsPage() {
  return (
    <SidebarLayout>
      <div className="p-6 space-y-6 max-w-[2560px] mx-auto">
        <ReportControls />
        <WeeklyDigest />
        <ReportList reports={REPORTS} />
      </div>
    </SidebarLayout>
  );
}
