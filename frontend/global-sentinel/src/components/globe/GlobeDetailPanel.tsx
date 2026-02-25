"use client";

import Link from "next/link";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CausalChainTimeline } from "./CausalChainTimeline";
import { RiskDriverBars } from "./RiskDriverBars";
import type { CountryData } from "@/lib/types";

type Props = { country: CountryData | null; onClose: () => void };

export function GlobeDetailPanel({ country, onClose }: Props) {
  if (!country) return null;

  const { subScores, confidence, mlMetadata, briefText, causalChain, headlines, featureImportance, forecast } = country;

  return (
    <div className="w-[400px] flex-shrink-0 border-l border-border bg-card flex flex-col h-full transition-all duration-300 overflow-hidden">
      <div className="flex items-start justify-between p-4 border-b border-border shrink-0">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <span>{country.flag}</span>
            <span>{country.name}</span>
          </h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl font-bold tabular-nums">{country.riskScore}</span>
            <Badge variant={country.riskLevel === "HIGH" || country.riskLevel === "CRITICAL" ? "destructive" : "secondary"}>
              {country.riskLevel}
            </Badge>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close panel">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          {/* Sub-scores */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Sub-scores</h3>
            <div className="space-y-2">
              {[
                { label: "Conflict Intensity", value: subScores.conflictIntensity },
                { label: "Social Unrest", value: subScores.socialUnrest },
                { label: "Economic Stress", value: subScores.economicStress },
              ].map(({ label, value }) => (
                <div key={label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>{label}</span>
                    <span>{value}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div className="h-full bg-primary rounded-full" style={{ width: `${value}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Confidence + data sources */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Confidence</h3>
            <div className="h-2 rounded-full bg-muted overflow-hidden mb-2">
              <div className="h-full bg-primary rounded-full" style={{ width: `${confidence * 100}%` }} />
            </div>
            <div className="flex flex-wrap gap-1">
              {mlMetadata.dataSources.map((src) => (
                <Badge key={src} variant="outline" className="text-[10px]">
                  {src}
                </Badge>
              ))}
            </div>
          </section>

          {/* Intelligence Brief */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Analysis</h3>
            <div className="border-l-4 border-primary pl-3 space-y-2">
              {briefText.map((p, i) => (
                <p key={i} className="text-sm">{p}</p>
              ))}
            </div>
          </section>

          {/* Causal Chain */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Causal Chain</h3>
            <CausalChainTimeline steps={causalChain} />
          </section>

          {/* ML Risk Drivers */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">ML Risk Drivers</h3>
            <RiskDriverBars drivers={featureImportance} />
          </section>

          {/* Media Sentiment */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Media Sentiment</h3>
            <div className="h-3 rounded-full overflow-hidden flex">
              <div className="bg-red-500/80" style={{ width: "33%" }} />
              <div className="bg-muted-foreground/50" style={{ width: "34%" }} />
              <div className="bg-green-500/80" style={{ width: "33%" }} />
            </div>
            <ul className="mt-2 space-y-1">
              {headlines.slice(0, 3).map((h, i) => (
                <li key={i} className="flex items-center gap-2 text-xs">
                  <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                    h.sentiment === "negative" ? "bg-red-500" : h.sentiment === "positive" ? "bg-green-500" : "bg-muted-foreground"
                  }`} />
                  <span className="truncate">{h.text}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* 90-Day Forecast */}
          <section>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground mb-2">90-Day Forecast</h3>
            <div className="flex gap-4">
              <div className="text-center">
                <div className="text-lg font-bold tabular-nums">{forecast.score30d}</div>
                <div className="text-[10px] text-muted-foreground">30d</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold tabular-nums">{forecast.score60d}</div>
                <div className="text-[10px] text-muted-foreground">60d</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold tabular-nums">{forecast.score90d}</div>
                <div className="text-[10px] text-muted-foreground">90d</div>
              </div>
            </div>
            <Badge variant="outline" className="mt-2 text-xs">{forecast.trend}</Badge>
          </section>

          {/* Action row */}
          <div className="pt-2">
            <Link
              href={`/country/${country.code}`}
              className="text-sm font-medium text-primary hover:underline"
            >
              Open Full Intel →
            </Link>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
