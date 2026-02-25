/**
 * Threat Feed — placeholder event stream data.
 */

export interface ThreatFeedEvent {
  id: string;
  timestamp: string;
  country: string;
  countryCode: string;
  flag: string;
  eventType: "ANOMALY" | "ESCALATION" | "DE-ESCALATION" | "NEW INTEL";
  severity: string;
  summary: string;
  detail: string;
  relatedHeadlines: string[];
}

export const THREAT_FEED_EVENTS: ThreatFeedEvent[] = [
  { id: "1", timestamp: "2 min ago", country: "Ukraine", countryCode: "UA", flag: "🇺🇦", eventType: "ANOMALY", severity: "HIGH", summary: "PLACEHOLDER: Risk score spike detected in Donbas region.", detail: "PLACEHOLDER detail text for anomaly event.", relatedHeadlines: ["PLACEHOLDER headline 1", "PLACEHOLDER headline 2"] },
  { id: "2", timestamp: "15 min ago", country: "Iran", countryCode: "IR", flag: "🇮🇷", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Military activity increase along border.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "3", timestamp: "32 min ago", country: "Ethiopia", countryCode: "ET", flag: "🇪🇹", eventType: "DE-ESCALATION", severity: "MEDIUM", summary: "PLACEHOLDER: Ceasefire talks progress reported.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "4", timestamp: "1 hour ago", country: "Pakistan", countryCode: "PK", flag: "🇵🇰", eventType: "NEW INTEL", severity: "MEDIUM", summary: "PLACEHOLDER: New intelligence report on regional stability.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "5", timestamp: "1 hour ago", country: "Sudan", countryCode: "SD", flag: "🇸🇩", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Conflict intensity rise in Darfur.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "6", timestamp: "2 hours ago", country: "Taiwan", countryCode: "TW", flag: "🇹🇼", eventType: "ANOMALY", severity: "MEDIUM", summary: "PLACEHOLDER: Unusual naval activity detected.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "7", timestamp: "2 hours ago", country: "Venezuela", countryCode: "VE", flag: "🇻🇪", eventType: "DE-ESCALATION", severity: "LOW", summary: "PLACEHOLDER: Protest activity decline.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "8", timestamp: "3 hours ago", country: "Myanmar", countryCode: "MM", flag: "🇲🇲", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Clashes in northern regions.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "9", timestamp: "3 hours ago", country: "Ukraine", countryCode: "UA", flag: "🇺🇦", eventType: "NEW INTEL", severity: "MEDIUM", summary: "PLACEHOLDER: Updated assessment from GDELT.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "10", timestamp: "4 hours ago", country: "Yemen", countryCode: "YE", flag: "🇾🇪", eventType: "DE-ESCALATION", severity: "MEDIUM", summary: "PLACEHOLDER: Humanitarian corridor opened.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "11", timestamp: "4 hours ago", country: "Iran", countryCode: "IR", flag: "🇮🇷", eventType: "ANOMALY", severity: "HIGH", summary: "PLACEHOLDER: Sentiment shift in state media.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "12", timestamp: "5 hours ago", country: "Syria", countryCode: "SY", flag: "🇸🇾", eventType: "ESCALATION", severity: "CRITICAL", summary: "PLACEHOLDER: Idlib escalation.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "13", timestamp: "5 hours ago", country: "Brazil", countryCode: "BR", flag: "🇧🇷", eventType: "NEW INTEL", severity: "LOW", summary: "PLACEHOLDER: Economic indicators update.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "14", timestamp: "5 hours ago", country: "Serbia", countryCode: "RS", flag: "🇷🇸", eventType: "DE-ESCALATION", severity: "LOW", summary: "PLACEHOLDER: Diplomatic dialogue progress.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "15", timestamp: "6 hours ago", country: "Libya", countryCode: "LY", flag: "🇱🇾", eventType: "ANOMALY", severity: "MEDIUM", summary: "PLACEHOLDER: Oil facility disruption.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "16", timestamp: "6 hours ago", country: "Iraq", countryCode: "IQ", flag: "🇮🇶", eventType: "NEW INTEL", severity: "MEDIUM", summary: "PLACEHOLDER: Security force deployment.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "17", timestamp: "6 hours ago", country: "Afghanistan", countryCode: "AF", flag: "🇦🇫", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Border incident.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "18", timestamp: "6 hours ago", country: "Colombia", countryCode: "CO", flag: "🇨🇴", eventType: "DE-ESCALATION", severity: "LOW", summary: "PLACEHOLDER: Peace process update.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "19", timestamp: "6 hours ago", country: "Nigeria", countryCode: "NG", flag: "🇳🇬", eventType: "ANOMALY", severity: "MEDIUM", summary: "PLACEHOLDER: Northeast activity spike.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "20", timestamp: "6 hours ago", country: "Haiti", countryCode: "HT", flag: "🇭🇹", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Capital unrest.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "21", timestamp: "6 hours ago", country: "DRC", countryCode: "CD", flag: "🇨🇩", eventType: "NEW INTEL", severity: "MEDIUM", summary: "PLACEHOLDER: Eastern region assessment.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
  { id: "22", timestamp: "6 hours ago", country: "Mali", countryCode: "ML", flag: "🇲🇱", eventType: "ESCALATION", severity: "HIGH", summary: "PLACEHOLDER: Sahel security update.", detail: "PLACEHOLDER detail.", relatedHeadlines: [] },
];
