"use client";

import { useState, useCallback } from "react";
import { WATCHLIST_COUNTRIES } from "@/lib/placeholder-data";
import dynamic from "next/dynamic";
import { GlobeMap } from "@/components/globe/GlobeMap";
import { GlobeOverlay } from "@/components/globe/GlobeOverlay";
import { GlobeDetailPanel } from "@/components/globe/GlobeDetailPanel";
import { GlobeAnalyticsStrip } from "@/components/globe/GlobeAnalyticsStrip";

const GlobeMapClient = dynamic(() => Promise.resolve(GlobeMap), { ssr: false });

export default function GlobePage() {
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const selectedCountry = selectedCode
    ? WATCHLIST_COUNTRIES.find((c) => c.code === selectedCode) ?? null
    : null;

  const handleCountrySelect = useCallback((code: string) => setSelectedCode(code), []);
  const handleClose = useCallback(() => setSelectedCode(null), []);

  return (
    <div className="relative flex h-[calc(100vh-52px)] w-full overflow-hidden">
      <div className={`relative flex-1 min-w-0 transition-all duration-300 ${selectedCountry ? "mr-0" : ""}`}>
        <GlobeMapClient onCountrySelect={handleCountrySelect} />
        <GlobeOverlay onCountrySelect={handleCountrySelect} selectedCode={selectedCode} />
        <div className="absolute top-4 right-4 z-10 rounded-md bg-white/95 backdrop-blur-sm border border-border px-2 py-1 text-xs text-muted-foreground">
          Live · Updated 2 min ago
        </div>
        <div className="absolute bottom-24 left-4 z-10 rounded-md bg-white/95 backdrop-blur-sm border border-border p-2 text-xs">
          <p className="font-semibold text-muted-foreground mb-1.5">Risk level</p>
          <div className="flex flex-wrap gap-2">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-green-500" /> LOW
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-yellow-500" /> MOD
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-orange-500" /> ELEV
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-red-500" /> HIGH
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-red-700" /> CRIT
            </span>
          </div>
        </div>
        <GlobeAnalyticsStrip />
      </div>
      {selectedCountry && (
        <GlobeDetailPanel country={selectedCountry} onClose={handleClose} />
      )}
    </div>
  );
}
