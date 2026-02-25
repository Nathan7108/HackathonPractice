"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import type { CountryData } from "@/lib/types";

type Props = { country: CountryData };

export function CountryHeader({ country }: Props) {
  return (
    <header className="sticky top-[52px] z-20 flex items-center justify-between gap-4 p-4 bg-card border-b border-border shadow-sm">
      <div className="flex items-center gap-4 min-w-0">
        <span className="text-4xl shrink-0" aria-hidden>{country.flag}</span>
        <div>
          <h1 className="text-2xl font-semibold truncate">{country.name}</h1>
          <p className="text-sm text-muted-foreground">
            Eastern Europe · 44M population · GDP $160B · Semi-presidential republic (PLACEHOLDER)
          </p>
        </div>
        <div className="flex items-baseline gap-2 shrink-0">
          <span className="text-4xl font-bold tabular-nums text-primary">{country.riskScore}</span>
          <Badge variant={country.riskLevel === "HIGH" || country.riskLevel === "CRITICAL" ? "destructive" : "secondary"}>
            {country.riskLevel}
          </Badge>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Button variant="outline" size="sm">Export Report</Button>
        <Button variant="outline" size="sm">Add to Watchlist</Button>
        <Button variant="outline" size="sm">Set Alert</Button>
        <Select defaultValue={country.code}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Switch country" />
          </SelectTrigger>
          <SelectContent>
            {WATCHLIST_COUNTRIES.map((c) => (
              <SelectItem key={c.code} value={c.code}>
                {c.flag} {c.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </header>
  );
}
