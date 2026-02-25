import { ThreatFilters } from "@/components/threats/ThreatFilters";
import { ThreatFeedList } from "@/components/threats/ThreatFeedList";
import { ThreatSidebar } from "@/components/threats/ThreatSidebar";
import { THREAT_FEED_EVENTS } from "@/lib/threat-feed-data";

export default function ThreatFeedPage() {
  return (
    <div className="p-6 flex gap-6 max-w-[2560px] mx-auto">
      <div className="flex-1 min-w-0 space-y-4" style={{ maxWidth: "65%" }}>
        <ThreatFilters />
        <ThreatFeedList events={THREAT_FEED_EVENTS} />
      </div>
      <ThreatSidebar />
    </div>
  );
}
