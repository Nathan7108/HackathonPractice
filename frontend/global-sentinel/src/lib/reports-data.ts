/**
 * Reports tab — placeholder intelligence brief library.
 */

export interface ReportEntry {
  id: string;
  country: string;
  countryCode: string;
  flag: string;
  generatedAt: string;
  riskScore: number;
  riskLevel: string;
  briefPreview: string[];
  fullBrief: string;
  status: "complete" | "generating";
}

export const REPORTS: ReportEntry[] = [
  { id: "1", country: "Ukraine", countryCode: "UA", flag: "🇺🇦", generatedAt: "2025-02-24", riskScore: 87, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: First line of brief summary.", "PLACEHOLDER: Second line of brief summary."], fullBrief: "PLACEHOLDER: Full intelligence brief text for Ukraine. Replace with real generated content in production. This section can be multiple paragraphs.", status: "complete" },
  { id: "2", country: "Iran", countryCode: "IR", flag: "🇮🇷", generatedAt: "2025-02-24", riskScore: 79, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: Iran brief line 1.", "PLACEHOLDER: Iran brief line 2."], fullBrief: "PLACEHOLDER: Full brief for Iran.", status: "complete" },
  { id: "3", country: "Pakistan", countryCode: "PK", flag: "🇵🇰", generatedAt: "2025-02-23", riskScore: 63, riskLevel: "ELEVATED", briefPreview: ["PLACEHOLDER: Pakistan line 1.", "PLACEHOLDER: Pakistan line 2."], fullBrief: "PLACEHOLDER: Full brief for Pakistan.", status: "complete" },
  { id: "4", country: "Ethiopia", countryCode: "ET", flag: "🇪🇹", generatedAt: "2025-02-23", riskScore: 58, riskLevel: "ELEVATED", briefPreview: ["PLACEHOLDER: Ethiopia line 1.", "PLACEHOLDER: Ethiopia line 2."], fullBrief: "PLACEHOLDER: Full brief for Ethiopia.", status: "complete" },
  { id: "5", country: "Sudan", countryCode: "SD", flag: "🇸🇩", generatedAt: "2025-02-22", riskScore: 82, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: Sudan line 1.", "PLACEHOLDER: Sudan line 2."], fullBrief: "PLACEHOLDER: Full brief for Sudan.", status: "complete" },
  { id: "6", country: "Syria", countryCode: "SY", flag: "🇸🇾", generatedAt: "2025-02-22", riskScore: 91, riskLevel: "CRITICAL", briefPreview: ["PLACEHOLDER: Syria line 1.", "PLACEHOLDER: Syria line 2."], fullBrief: "PLACEHOLDER: Full brief for Syria.", status: "complete" },
  { id: "7", country: "Venezuela", countryCode: "VE", flag: "🇻🇪", generatedAt: "2025-02-21", riskScore: 41, riskLevel: "MODERATE", briefPreview: ["PLACEHOLDER: Venezuela line 1.", "PLACEHOLDER: Venezuela line 2."], fullBrief: "PLACEHOLDER: Full brief for Venezuela.", status: "complete" },
  { id: "8", country: "Taiwan", countryCode: "TW", flag: "🇹🇼", generatedAt: "2025-02-21", riskScore: 38, riskLevel: "MODERATE", briefPreview: ["PLACEHOLDER: Taiwan line 1.", "PLACEHOLDER: Taiwan line 2."], fullBrief: "PLACEHOLDER: Full brief for Taiwan.", status: "complete" },
  { id: "9", country: "Myanmar", countryCode: "MM", flag: "🇲🇲", generatedAt: "2025-02-20", riskScore: 76, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: Myanmar line 1.", "PLACEHOLDER: Myanmar line 2."], fullBrief: "PLACEHOLDER: Full brief for Myanmar.", status: "complete" },
  { id: "10", country: "Yemen", countryCode: "YE", flag: "🇾🇪", generatedAt: "2025-02-20", riskScore: 88, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: Yemen line 1.", "PLACEHOLDER: Yemen line 2."], fullBrief: "PLACEHOLDER: Full brief for Yemen.", status: "complete" },
  { id: "11", country: "Afghanistan", countryCode: "AF", flag: "🇦🇫", generatedAt: "2025-02-19", riskScore: 85, riskLevel: "HIGH", briefPreview: ["PLACEHOLDER: Afghanistan line 1.", "PLACEHOLDER: Afghanistan line 2."], fullBrief: "PLACEHOLDER: Full brief for Afghanistan.", status: "complete" },
];
