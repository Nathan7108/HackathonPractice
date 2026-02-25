"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { ThreatFeedEvent } from "@/lib/threat-feed-data";
import { cn } from "@/lib/utils";

const BADGE_CLASS: Record<string, string> = {
  ANOMALY: "bg-red-500/90 text-white",
  ESCALATION: "bg-orange-500/90 text-white",
  "DE-ESCALATION": "bg-green-500/90 text-white",
  "NEW INTEL": "bg-blue-500/90 text-white",
};

type Props = { event: ThreatFeedEvent };

export function ThreatFeedCard({ event }: Props) {
  const [expanded, setExpanded] = useState(false);
  const badgeClass = BADGE_CLASS[event.eventType] ?? "bg-muted";
  return (
    <Card>
      <CardHeader className="py-3 px-4 flex flex-row items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-muted-foreground shrink-0">{event.timestamp}</span>
          <span className="text-lg shrink-0">{event.flag}</span>
          <span className="font-medium truncate">{event.country}</span>
        </div>
        <span className={cn("text-[10px] font-semibold px-1.5 py-0.5 rounded shrink-0", badgeClass)}>
          {event.eventType}
        </span>
      </CardHeader>
      <CardContent className="pt-0 px-4 pb-3">
        <p className="text-sm">{event.summary}</p>
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? "Less" : "More"}
        </button>
        {expanded && (
          <div className="mt-2 pt-2 border-t border-border text-xs text-muted-foreground">
            <p>{event.detail}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
