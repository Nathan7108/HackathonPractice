"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { CountryData } from "@/lib/types";

type Props = { country: CountryData };

export function KeyActors({ country }: Props) {
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold uppercase text-muted-foreground">Key Actors</h2>
      <div className="grid gap-2 sm:grid-cols-2">
        {country.keyActors.map((actor) => (
          <Card key={actor.name}>
            <CardContent className="p-3">
              <p className="font-medium text-sm">{actor.name}</p>
              <p className="text-xs text-muted-foreground">{actor.role}</p>
              <Badge variant="outline" className="mt-1 text-xs">{actor.stance}</Badge>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
