"use client";

import { ThreatFeedCard } from "./ThreatFeedCard";
import type { ThreatFeedEvent } from "@/lib/threat-feed-data";

type Props = { events: ThreatFeedEvent[] };

export function ThreatFeedList({ events }: Props) {
  return (
    <div className="space-y-3 overflow-y-auto pr-2">
      {events.map((event) => (
        <ThreatFeedCard key={event.id} event={event} />
      ))}
    </div>
  );
}
