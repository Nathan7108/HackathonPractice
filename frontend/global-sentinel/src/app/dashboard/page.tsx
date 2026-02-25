import { KpiCardRow } from "@/components/dashboard/KpiCardRow";
import { PanelHeader } from "@/components/dashboard/PanelHeader";
import { RiskDistributionChart } from "@/components/dashboard/RiskDistributionChart";
import { RegionalBreakdown } from "@/components/dashboard/RegionalBreakdown";
import { TopRiskTable } from "@/components/dashboard/TopRiskTable";
import { EscalationLists } from "@/components/dashboard/EscalationLists";
import { SentimentTrendChart } from "@/components/dashboard/SentimentTrendChart";
import { ModelPerformanceChart } from "@/components/dashboard/ModelPerformanceChart";
import { Card, CardContent } from "@/components/ui/card";

export default function DashboardPage() {
  return (
    <div className="p-2 max-w-[2560px] mx-auto">
      {/* KPI row: 5 cards across, equal width, gap-0.5 */}
      <div className="mb-0.5">
        <KpiCardRow />
      </div>

      {/* Main grid: 5 cols on lg, gap-0.5, panels fill cells */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-0.5">
        {/* Row 1: Heatmap (3) | Top 10 Table (2) */}
        <Card className="lg:col-span-3 min-h-0 p-0 border border-border rounded-sm shadow-none flex flex-col">
          <PanelHeader title="Global Risk Heatmap" />
          <CardContent className="p-3 flex-1 min-h-0">
            <div className="h-[120px] rounded-sm bg-gradient-to-b from-gray-100 to-gray-50 dark:from-gray-800/50 dark:to-gray-900/40 flex items-center justify-center text-gray-500 dark:text-gray-400 text-xs font-medium">
              LIVE MAP — INTEGRATION PENDING
            </div>
          </CardContent>
        </Card>
        <div className="lg:col-span-2 min-h-0">
          <TopRiskTable />
        </div>

        {/* Row 2: Risk Distribution (3) | Escalation Lists (2) */}
        <div className="lg:col-span-3 min-h-0">
          <RiskDistributionChart />
        </div>
        <div className="lg:col-span-2 min-h-0">
          <EscalationLists />
        </div>

        {/* Row 3: Regional Breakdown (3) | Model Performance (2) */}
        <div className="lg:col-span-3 min-h-0">
          <RegionalBreakdown />
        </div>
        <div className="lg:col-span-2 min-h-0">
          <ModelPerformanceChart />
        </div>

        {/* Row 4: Sentiment full width (5) */}
        <div className="lg:col-span-5 min-h-0">
          <SentimentTrendChart />
        </div>
      </div>
    </div>
  );
}
