"use client";

const NEIGHBORS = [
  { name: "Moldova", flag: "🇲🇩", correlation: 0.82, score: 45 },
  { name: "Poland", flag: "🇵🇱", correlation: 0.61, score: 22 },
  { name: "Romania", flag: "🇷🇴", correlation: 0.58, score: 28 },
  { name: "Belarus", flag: "🇧🇾", correlation: 0.54, score: 65 },
  { name: "Hungary", flag: "🇭🇺", correlation: 0.49, score: 18 },
];

export function NeighborCorrelation() {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Neighboring Country Correlation</h2>
      <div className="space-y-2">
        {NEIGHBORS.map((n) => (
          <div key={n.name} className="flex items-center justify-between rounded-md border border-border p-2">
            <span>{n.flag} {n.name}</span>
            <span className="text-xs text-muted-foreground">+{n.correlation.toFixed(2)}</span>
            <span className="text-sm font-mono">{n.score}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
