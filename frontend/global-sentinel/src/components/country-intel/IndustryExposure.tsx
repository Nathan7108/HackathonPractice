"use client";

import { Badge } from "@/components/ui/badge";
import type { CountryData } from "@/lib/types";

type Props = { country: CountryData };

export function IndustryExposure({ country }: Props) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Industry Exposure</h2>
      <ul className="space-y-2">
        {country.industryExposure.map((ind) => (
          <li key={ind.industry} className="flex items-center justify-between">
            <span className="text-sm">{ind.industry}</span>
            <Badge variant={ind.impactLevel === "HIGH" ? "destructive" : "secondary"} className="text-xs">
              {ind.impactLevel}
            </Badge>
          </li>
        ))}
      </ul>
    </section>
  );
}
