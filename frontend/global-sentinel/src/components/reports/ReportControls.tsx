"use client";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";

export function ReportControls() {
  return (
    <div className="flex flex-wrap items-center gap-4 p-4 border-b border-border bg-card rounded-lg">
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input type="search" placeholder="Search reports..." className="pl-8" aria-label="Search reports" />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Date range</span>
        <input type="date" className="h-9 rounded-md border border-input bg-background px-3 text-sm" aria-label="From" />
        <span className="text-muted-foreground">–</span>
        <input type="date" className="h-9 rounded-md border border-input bg-background px-3 text-sm" aria-label="To" />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Country</span>
        <select className="h-9 rounded-md border border-input bg-background px-3 text-sm" aria-label="Country filter">
          <option>All countries</option>
        </select>
      </div>
      <Button>Generate New Brief</Button>
    </div>
  );
}
