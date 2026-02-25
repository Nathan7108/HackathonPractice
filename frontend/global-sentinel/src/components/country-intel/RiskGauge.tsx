"use client";

type Props = { score: number; riskLevel: string };

export function RiskGauge({ score, riskLevel }: Props) {
  const pct = Math.min(100, Math.max(0, score));
  const rotation = -90 + (pct / 100) * 180;
  const color =
    riskLevel === "CRITICAL" ? "#450a0a" :
    riskLevel === "HIGH" ? "#dc2626" :
    riskLevel === "ELEVATED" ? "#ea580c" :
    riskLevel === "MODERATE" ? "#eab308" : "#22c55e";
  const r = 60;
  const cx = 80;
  const cy = 80;
  const start = { x: cx - r, y: cy };
  const endAngle = (rotation * Math.PI) / 180;
  const end = { x: cx + r * Math.cos(endAngle), y: cy + r * Math.sin(endAngle) };
  const largeArc = pct > 50 ? 1 : 0;
  const pathD = `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Risk Assessment</h2>
      <div className="relative w-40 h-28">
        <svg viewBox="0 0 160 100" className="w-full h-full">
          <path
            d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth="12"
            strokeLinecap="round"
          />
          <path
            d={pathD}
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center pt-6">
          <span className="text-2xl font-bold tabular-nums">{score}</span>
        </div>
      </div>
    </section>
  );
}
