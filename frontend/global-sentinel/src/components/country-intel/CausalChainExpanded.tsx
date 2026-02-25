"use client";

import type { CausalChainStep } from "@/lib/types";

const STEP_LABELS = ["Signal", "Context", "Escalation", "Trigger", "Impact", "Response", "Outcome"];
const GRADIENT_STOPS = ["#34a853", "#f9a825", "#ea580c", "#dc2626", "#b91c1c", "#8b0000", "#450a0a"];

type Props = { steps: CausalChainStep[] };

export function CausalChainExpanded({ steps }: Props) {
  const displaySteps = steps.slice(0, 7).map((s, i) => ({
    label: STEP_LABELS[i] ?? s.label,
    description: s.description,
    color: GRADIENT_STOPS[i],
  }));
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Causal Chain</h2>
      <div className="relative pl-6">
        <div
          className="absolute left-[11px] top-2 bottom-2 w-0.5 rounded-full"
          style={{
            background: `linear-gradient(to bottom, ${GRADIENT_STOPS.join(", ")})`,
          }}
        />
        {displaySteps.map((step, i) => (
          <div key={i} className="relative flex gap-4 py-4 min-h-[80px]">
            <div
              className="absolute left-0 w-6 h-6 rounded-full border-2 border-background flex items-center justify-center text-xs font-bold shrink-0 z-10"
              style={{ backgroundColor: step.color, color: "white" }}
            >
              {i + 1}
            </div>
            <div className="pl-2">
              <p className="font-semibold text-sm">{step.label}</p>
              <p className="text-sm text-muted-foreground mt-1">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
