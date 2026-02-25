/**
 * Analytics tab — extended placeholder data for table and charts.
 */

export interface CountryTableRow {
  flag: string;
  country: string;
  region: string;
  riskScore: number;
  riskLevel: string;
  confidence: number;
  anomaly: boolean;
  change7d: number;
  change30d: number;
  battles: number;
  fatalities: number;
  sentiment: string;
}

export const ALL_COUNTRIES_TABLE: CountryTableRow[] = [
  { flag: "🇺🇦", country: "Ukraine", region: "Europe", riskScore: 87, riskLevel: "HIGH", confidence: 0.92, anomaly: true, change7d: 2, change30d: 5, battles: 47, fatalities: 312, sentiment: "negative" },
  { flag: "🇮🇷", country: "Iran", region: "Middle East", riskScore: 79, riskLevel: "HIGH", confidence: 0.89, anomaly: true, change7d: 1, change30d: 6, battles: 12, fatalities: 89, sentiment: "negative" },
  { flag: "🇵🇰", country: "Pakistan", region: "Asia", riskScore: 63, riskLevel: "ELEVATED", confidence: 0.85, anomaly: true, change7d: 2, change30d: 5, battles: 28, fatalities: 156, sentiment: "neutral" },
  { flag: "🇪🇹", country: "Ethiopia", region: "Africa", riskScore: 58, riskLevel: "ELEVATED", confidence: 0.81, anomaly: true, change7d: -1, change30d: -2, battles: 34, fatalities: 201, sentiment: "neutral" },
  { flag: "🇻🇪", country: "Venezuela", region: "Americas", riskScore: 41, riskLevel: "MODERATE", confidence: 0.78, anomaly: false, change7d: 0, change30d: 1, battles: 8, fatalities: 23, sentiment: "neutral" },
  { flag: "🇹🇼", country: "Taiwan", region: "Asia Pacific", riskScore: 38, riskLevel: "MODERATE", confidence: 0.82, anomaly: true, change7d: 1, change30d: 3, battles: 0, fatalities: 0, sentiment: "neutral" },
  { flag: "🇷🇸", country: "Serbia", region: "Europe", riskScore: 18, riskLevel: "LOW", confidence: 0.88, anomaly: false, change7d: 0, change30d: 0, battles: 2, fatalities: 5, sentiment: "positive" },
  { flag: "🇧🇷", country: "Brazil", region: "Americas", riskScore: 12, riskLevel: "LOW", confidence: 0.9, anomaly: false, change7d: 0, change30d: -1, battles: 5, fatalities: 12, sentiment: "neutral" },
  { flag: "🇸🇩", country: "Sudan", region: "Africa", riskScore: 82, riskLevel: "HIGH", confidence: 0.86, anomaly: true, change7d: 14, change30d: 18, battles: 56, fatalities: 420, sentiment: "negative" },
  { flag: "🇲🇲", country: "Myanmar", region: "Asia", riskScore: 76, riskLevel: "HIGH", confidence: 0.84, anomaly: true, change7d: 11, change30d: 12, battles: 41, fatalities: 287, sentiment: "negative" },
  { flag: "🇸🇾", country: "Syria", region: "Middle East", riskScore: 91, riskLevel: "CRITICAL", confidence: 0.93, anomaly: true, change7d: 3, change30d: 4, battles: 62, fatalities: 389, sentiment: "negative" },
  { flag: "🇾🇪", country: "Yemen", region: "Middle East", riskScore: 88, riskLevel: "HIGH", confidence: 0.9, anomaly: true, change7d: -3, change30d: -5, battles: 38, fatalities: 234, sentiment: "neutral" },
  { flag: "🇸🇴", country: "Somalia", region: "Africa", riskScore: 74, riskLevel: "HIGH", confidence: 0.83, anomaly: true, change7d: 2, change30d: 4, battles: 29, fatalities: 178, sentiment: "negative" },
  { flag: "🇨🇩", country: "DRC", region: "Africa", riskScore: 69, riskLevel: "ELEVATED", confidence: 0.82, anomaly: true, change7d: 4, change30d: 7, battles: 44, fatalities: 267, sentiment: "negative" },
  { flag: "🇱🇾", country: "Libya", region: "Africa", riskScore: 64, riskLevel: "ELEVATED", confidence: 0.8, anomaly: true, change7d: -4, change30d: -6, battles: 19, fatalities: 98, sentiment: "neutral" },
  { flag: "🇮🇶", country: "Iraq", region: "Middle East", riskScore: 59, riskLevel: "ELEVATED", confidence: 0.85, anomaly: false, change7d: -2, change30d: -5, battles: 22, fatalities: 112, sentiment: "neutral" },
  { flag: "🇦🇫", country: "Afghanistan", region: "Asia", riskScore: 85, riskLevel: "HIGH", confidence: 0.91, anomaly: true, change7d: 1, change30d: 2, battles: 51, fatalities: 334, sentiment: "negative" },
  { flag: "🇲🇱", country: "Mali", region: "Africa", riskScore: 71, riskLevel: "HIGH", confidence: 0.81, anomaly: true, change7d: 5, change30d: 8, battles: 33, fatalities: 189, sentiment: "negative" },
  { flag: "🇳🇪", country: "Niger", region: "Africa", riskScore: 66, riskLevel: "ELEVATED", confidence: 0.79, anomaly: true, change7d: 6, change30d: 9, battles: 26, fatalities: 145, sentiment: "negative" },
  { flag: "🇳🇬", country: "Nigeria", region: "Africa", riskScore: 54, riskLevel: "ELEVATED", confidence: 0.77, anomaly: false, change7d: 2, change30d: 3, battles: 31, fatalities: 167, sentiment: "neutral" },
  { flag: "🇭🇹", country: "Haiti", region: "Americas", riskScore: 72, riskLevel: "HIGH", confidence: 0.78, anomaly: true, change7d: 9, change30d: 11, battles: 15, fatalities: 78, sentiment: "negative" },
  { flag: "🇨🇴", country: "Colombia", region: "Americas", riskScore: 45, riskLevel: "MODERATE", confidence: 0.82, anomaly: false, change7d: -8, change30d: -10, battles: 11, fatalities: 34, sentiment: "positive" },
];

export const RISK_HISTOGRAM = [
  { range: "0-10", count: 42 },
  { range: "11-20", count: 35 },
  { range: "21-30", count: 28 },
  { range: "31-40", count: 24 },
  { range: "41-50", count: 22 },
  { range: "51-60", count: 18 },
  { range: "61-70", count: 15 },
  { range: "71-80", count: 12 },
  { range: "81-90", count: 8 },
  { range: "91-100", count: 6 },
];

export const SCATTER_DATA = [
  { country: "Ukraine", gdpGrowth: -2.1, riskScore: 87, riskLevel: "HIGH" },
  { country: "Iran", gdpGrowth: 1.2, riskScore: 79, riskLevel: "HIGH" },
  { country: "Pakistan", gdpGrowth: 2.5, riskScore: 63, riskLevel: "ELEVATED" },
  { country: "Ethiopia", gdpGrowth: 4.1, riskScore: 58, riskLevel: "ELEVATED" },
  { country: "Venezuela", gdpGrowth: -5.2, riskScore: 41, riskLevel: "MODERATE" },
  { country: "Taiwan", gdpGrowth: 3.2, riskScore: 38, riskLevel: "MODERATE" },
  { country: "Serbia", gdpGrowth: 2.8, riskScore: 18, riskLevel: "LOW" },
  { country: "Brazil", gdpGrowth: 2.1, riskScore: 12, riskLevel: "LOW" },
  { country: "Sudan", gdpGrowth: -3.4, riskScore: 82, riskLevel: "HIGH" },
  { country: "Myanmar", gdpGrowth: 0.5, riskScore: 76, riskLevel: "HIGH" },
  { country: "Syria", gdpGrowth: -4.2, riskScore: 91, riskLevel: "CRITICAL" },
  { country: "Yemen", gdpGrowth: -1.8, riskScore: 88, riskLevel: "HIGH" },
  { country: "Somalia", gdpGrowth: 2.0, riskScore: 74, riskLevel: "HIGH" },
  { country: "DRC", gdpGrowth: 3.5, riskScore: 69, riskLevel: "ELEVATED" },
  { country: "Libya", gdpGrowth: 1.1, riskScore: 64, riskLevel: "ELEVATED" },
  { country: "Iraq", gdpGrowth: 2.9, riskScore: 59, riskLevel: "ELEVATED" },
  { country: "Afghanistan", gdpGrowth: -0.5, riskScore: 85, riskLevel: "HIGH" },
  { country: "Mali", gdpGrowth: 1.8, riskScore: 71, riskLevel: "HIGH" },
  { country: "Niger", gdpGrowth: 3.2, riskScore: 66, riskLevel: "ELEVATED" },
  { country: "Nigeria", gdpGrowth: 2.6, riskScore: 54, riskLevel: "ELEVATED" },
];

export const RISK_TRENDS: Record<string, { week: string; score: number }[]> = {
  ukraine: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 82 + Math.sin(i * 0.5) * 5 })),
  iran: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 76 + i * 0.3 })),
  pakistan: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 60 + Math.cos(i * 0.4) * 5 })),
  ethiopia: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 58 - i * 0.2 })),
  venezuela: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 40 + (i % 3) })),
  taiwan: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 36 + i * 0.2 })),
  serbia: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 18 })),
  brazil: Array.from({ length: 12 }, (_, i) => ({ week: `W${i + 1}`, score: 12 })),
};

export const CALENDAR_HEATMAP: { country: string; weeks: number[] }[] = [
  { country: "UA", weeks: [8, 7, 9, 8, 7, 8, 9, 8, 8, 7, 9, 8] },
  { country: "IR", weeks: [7, 7, 8, 7, 8, 7, 8, 8, 7, 8, 7, 8] },
  { country: "PK", weeks: [5, 6, 5, 6, 6, 5, 6, 6, 5, 6, 6, 5] },
  { country: "ET", weeks: [5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 5] },
  { country: "VE", weeks: [3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3] },
  { country: "TW", weeks: [3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3] },
  { country: "RS", weeks: [1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1] },
  { country: "BR", weeks: [1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1] },
];

export const SENTIMENT_VS_RISK = Array.from({ length: 12 }, (_, i) => ({
  week: `W${i + 1}`,
  sentiment: 40 + Math.sin(i * 0.6) * 15,
  riskScore: 45 + Math.cos(i * 0.4) * 20,
}));

export const FEATURE_BY_TIER = [
  { tier: "LOW", feature: "GDP growth", value: 28 },
  { tier: "LOW", feature: "Stability index", value: 22 },
  { tier: "LOW", feature: "Trade openness", value: 18 },
  { tier: "MODERATE", feature: "Unrest events", value: 32 },
  { tier: "MODERATE", feature: "Economic stress", value: 24 },
  { tier: "ELEVATED", feature: "Conflict intensity", value: 38 },
  { tier: "ELEVATED", feature: "Fatalities 30d", value: 30 },
  { tier: "HIGH", feature: "Battles 30d", value: 42 },
  { tier: "HIGH", feature: "Anomaly score", value: 35 },
  { tier: "CRITICAL", feature: "Geographic spread", value: 45 },
  { tier: "CRITICAL", feature: "Actor count", value: 40 },
];
