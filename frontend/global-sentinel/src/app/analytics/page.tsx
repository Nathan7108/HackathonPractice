import { SidebarLayout } from "@/components/layout/SidebarLayout";
import { AnalyticsControls } from "@/components/analytics/AnalyticsControls";
import { CountryRiskTable } from "@/components/analytics/CountryRiskTable";
import { RiskHistogram } from "@/components/analytics/RiskHistogram";
import { FeatureScatterPlot } from "@/components/analytics/FeatureScatterPlot";
import { RegionalComparison } from "@/components/analytics/RegionalComparison";
import { RiskTrendLines } from "@/components/analytics/RiskTrendLines";
import { ConflictHeatmap } from "@/components/analytics/ConflictHeatmap";
import { SentimentRiskCorrelation } from "@/components/analytics/SentimentRiskCorrelation";
import { FeatureByTierChart } from "@/components/analytics/FeatureByTierChart";

export default function AnalyticsPage() {
  return (
    <SidebarLayout>
      <div className="p-6 space-y-6 max-w-[2560px] mx-auto">
        <AnalyticsControls />
        <CountryRiskTable />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <RiskHistogram />
          <FeatureScatterPlot />
          <RegionalComparison />
          <RiskTrendLines />
          <ConflictHeatmap />
          <SentimentRiskCorrelation />
          <FeatureByTierChart />
        </div>
      </div>
    </SidebarLayout>
  );
}
