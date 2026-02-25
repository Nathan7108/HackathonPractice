"use client";

const METRICS = [
  { label: "Battles (30d)", value: "47" },
  { label: "Fatalities (30d)", value: "312" },
  { label: "Protest Events", value: "89" },
  { label: "Active Actors", value: "14" },
  { label: "Geographic Spread", value: "8 regions" },
  { label: "Anomaly Score", value: "0.72" },
  { label: "Civilian Events", value: "23" },
  { label: "Explosions", value: "31" },
  { label: "Event Accel.", value: "2.3x" },
];

export function KeyMetricsGrid() {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Key Metrics</h2>
      <div className="grid grid-cols-3 gap-2">
        {METRICS.map((m) => (
          <div key={m.label} className="rounded-md border border-border p-2">
            <p className="text-[10px] text-muted-foreground uppercase">{m.label}</p>
            <p className="text-lg font-bold tabular-nums">{m.value}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
