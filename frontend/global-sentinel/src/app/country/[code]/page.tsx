import { notFound } from "next/navigation";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import { CountryHeader } from "@/components/country-intel/CountryHeader";
import { IntelBrief } from "@/components/country-intel/IntelBrief";
import { CausalChainExpanded } from "@/components/country-intel/CausalChainExpanded";
import { IndustryExposure } from "@/components/country-intel/IndustryExposure";
import { KeyActors } from "@/components/country-intel/KeyActors";
import { RiskGauge } from "@/components/country-intel/RiskGauge";
import { KeyMetricsGrid } from "@/components/country-intel/KeyMetricsGrid";
import { HistoricalTimeline } from "@/components/country-intel/HistoricalTimeline";
import { ForecastChart } from "@/components/country-intel/ForecastChart";
import { SentimentTimeline } from "@/components/country-intel/SentimentTimeline";
import { MLTransparency } from "@/components/country-intel/MLTransparency";
import { NeighborCorrelation } from "@/components/country-intel/NeighborCorrelation";

type Props = { params: Promise<{ code: string }> };

export default async function CountryIntelPage({ params }: Props) {
  const { code } = await params;
  const country = WATCHLIST_COUNTRIES.find((c) => c.code.toUpperCase() === code.toUpperCase());
  if (!country) notFound();

  return (
    <div className="min-h-screen bg-muted/30">
      <CountryHeader country={country} />
      <div className="max-w-[1600px] mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7 space-y-8">
            <IntelBrief country={country} />
            <CausalChainExpanded steps={country.causalChain} />
            <IndustryExposure country={country} />
            <KeyActors country={country} />
          </div>
          <div className="lg:col-span-5 space-y-8">
            <RiskGauge score={country.riskScore} riskLevel={country.riskLevel} />
            <KeyMetricsGrid />
            <HistoricalTimeline />
            <ForecastChart forecast={country.forecast} />
            <SentimentTimeline headlines={country.headlines} />
            <MLTransparency country={country} />
            <NeighborCorrelation />
          </div>
        </div>
      </div>
    </div>
  );
}
