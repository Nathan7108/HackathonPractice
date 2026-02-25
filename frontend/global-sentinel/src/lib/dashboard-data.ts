/**
 * Dashboard tab — placeholder data for KPI cards, charts, and tables.
 */

export const RISK_DISTRIBUTION = [
  { tier: "LOW", count: 42 },
  { tier: "MODERATE", count: 38 },
  { tier: "ELEVATED", count: 28 },
  { tier: "HIGH", count: 15 },
  { tier: "CRITICAL", count: 7 },
];

export const REGIONAL_BREAKDOWN = [
  { region: "Middle East", avgRisk: 67, anomalies: 4, escalations: 2 },
  { region: "Europe", avgRisk: 52, anomalies: 3, escalations: 1 },
  { region: "Asia Pacific", avgRisk: 41, anomalies: 2, escalations: 1 },
  { region: "Africa", avgRisk: 58, anomalies: 3, escalations: 2 },
  { region: "Americas", avgRisk: 35, anomalies: 1, escalations: 0 },
  { region: "Central Asia", avgRisk: 61, anomalies: 2, escalations: 1 },
];

const DAYS = 30;
export const SENTIMENT_TREND_30D = Array.from({ length: DAYS }, (_, i) => ({
  day: `Feb ${i + 1}`,
  escalatory: 25 + Math.sin(i * 0.3) * 10,
  neutral: 50 + Math.cos(i * 0.2) * 8,
  deescalatory: 25 - Math.sin(i * 0.25) * 8,
}));

export const MODEL_PERFORMANCE = [
  { week: "W1", accuracy: 97.2 },
  { week: "W2", accuracy: 97.5 },
  { week: "W3", accuracy: 97.1 },
  { week: "W4", accuracy: 97.8 },
  { week: "W5", accuracy: 98.0 },
  { week: "W6", accuracy: 97.6 },
  { week: "W7", accuracy: 98.1 },
  { week: "W8", accuracy: 97.9 },
  { week: "W9", accuracy: 98.2 },
  { week: "W10", accuracy: 97.7 },
  { week: "W11", accuracy: 98.0 },
  { week: "W12", accuracy: 98.3 },
];

export const TOP_ESCALATING = [
  { country: "Sudan", delta: 14 },
  { country: "Myanmar", delta: 11 },
  { country: "Haiti", delta: 9 },
  { country: "Iran", delta: 6 },
  { country: "Pakistan", delta: 5 },
];

export const TOP_DEESCALATING = [
  { country: "Colombia", delta: -8 },
  { country: "Ethiopia", delta: -6 },
  { country: "Iraq", delta: -5 },
  { country: "Libya", delta: -4 },
  { country: "Yemen", delta: -3 },
];

export const KPI_SPARKLINE_DATA = [
  [44, 45, 46, 45, 47, 46, 47],
  [10, 11, 12, 11, 12, 12, 12],
  [15, 16, 17, 16, 17, 17, 17],
  [3, 4, 4, 4, 4, 4, 4],
  [97, 97.2, 97.5, 97.8, 98, 98, 98],
];
