"use client";

import type { CausalChainStep } from "@/lib/types";

const STEP_LABELS = [
  "Signal",
  "Context",
  "Escalation",
  "Trigger",
  "Impact",
  "Response",
  "Outcome",
];

type Props = { steps: CausalChainStep[] };

export function CausalChainTimeline({ steps }: Props) {
  const displaySteps = steps.slice(0, 7).map((s, i) => ({
    label: STEP_LABELS[i] ?? s.label,
    description: s.description,
  }));
  return (
    <div className="relative pl-4 border-l-2 border-muted space-y-4">
      {displaySteps.map((step, i) => (
        <div key={i} className="relative -left-[21px]">
          <div className="absolute left-0 w-3 h-3 rounded-full bg-primary border-2 border-background" />
          <div className="pl-4">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {step.label}
            </p>
            <p className="text-sm mt-0.5">{step.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
