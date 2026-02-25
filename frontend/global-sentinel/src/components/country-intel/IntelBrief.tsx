"use client";

import { Badge } from "@/components/ui/badge";
import type { CountryData } from "@/lib/types";

type Props = { country: CountryData };

export function IntelBrief({ country }: Props) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground flex items-center gap-2">
        Intelligence Brief
        <Badge variant="outline" className="text-xs">GPT-4o</Badge>
      </h2>
      <div className="border-l-4 border-primary pl-4 space-y-3">
        {country.briefText.map((p, i) => (
          <p key={i} className="text-sm">{p}</p>
        ))}
      </div>
    </section>
  );
}
