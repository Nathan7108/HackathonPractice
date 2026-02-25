/**
 * Sentinel AI — TypeScript types for country risk, analytics, and threat feed.
 */

export type RiskLevel = "LOW" | "MODERATE" | "ELEVATED" | "HIGH" | "CRITICAL";

export interface SubScores {
  conflictIntensity: number;
  socialUnrest: number;
  economicStress: number;
}

export interface AnomalyAlert {
  detected: boolean;
  score: number;
  severity: string;
}

export interface CausalChainStep {
  label: string;
  description: string;
}

export interface HeadlineSentiment {
  text: string;
  sentiment: string;
}

export interface FeatureImportance {
  name: string;
  percentage: number;
}

export interface MLMetadata {
  topDrivers: string[];
  dataSources: string[];
  modelVersion: string;
}

export interface ForecastData {
  score30d: number;
  score60d: number;
  score90d: number;
  trend: string;
}

export interface IndustryExposure {
  industry: string;
  impactLevel: string;
}

export interface KeyActor {
  name: string;
  role: string;
  stance: string;
}

export interface CountryData {
  code: string;
  name: string;
  flag: string;
  riskScore: number;
  riskLevel: RiskLevel;
  confidence: number;
  subScores: SubScores;
  anomaly: AnomalyAlert;
  mlMetadata: MLMetadata;
  briefText: string[];
  causalChain: CausalChainStep[];
  headlines: HeadlineSentiment[];
  featureImportance: FeatureImportance[];
  forecast: ForecastData;
  industryExposure: IndustryExposure[];
  keyActors: KeyActor[];
}

export interface ThreatFeedItem {
  id: string;
  timestamp: string;
  countryCode: string;
  countryName: string;
  eventType: string;
  title: string;
  summary: string;
  severity: string;
  source: string;
  url?: string;
}
