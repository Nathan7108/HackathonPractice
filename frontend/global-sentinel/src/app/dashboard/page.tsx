import { SidebarLayout } from "@/components/layout/SidebarLayout";
import { KpiCardRow } from "@/components/dashboard/KpiCardRow";
import { RiskDistributionChart } from "@/components/dashboard/RiskDistributionChart";
import { RegionalBreakdown } from "@/components/dashboard/RegionalBreakdown";
import { TopRiskTable } from "@/components/dashboard/TopRiskTable";
import { EscalationLists } from "@/components/dashboard/EscalationLists";
import { SentimentTrendChart } from "@/components/dashboard/SentimentTrendChart";
import { ModelPerformanceChart } from "@/components/dashboard/ModelPerformanceChart";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function DashboardPage() {
  return (
    <SidebarLayout>
      <div className="p-6 space-y-6 max-w-[2560px] mx-auto">
        {/* KPI row */}
        <KpiCardRow />

        {/* Two columns: left ~60%, right ~40% */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 space-y-6">
            <Card>
              <CardHeader>
                <h3 className="text-sm font-semibold">Global Risk Heatmap</h3>
              </CardHeader>
              <CardContent>
                <div className="h-[200px] rounded-md bg-muted flex items-center justify-center text-muted-foreground text-sm">
                  MAP PLACEHOLDER — flat projection risk heatmap
                </div>
              </CardContent>
            </Card>
            <RiskDistributionChart />
            <RegionalBreakdown />
          </div>
          <div className="lg:col-span-2 space-y-6">
            <TopRiskTable />
            <EscalationLists />
          </div>
        </div>

        {/* Bottom row: full width charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SentimentTrendChart />
          <ModelPerformanceChart />
        </div>
      </div>
    </SidebarLayout>
  );
}
