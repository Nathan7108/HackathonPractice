"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Globe, Bell, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export function TopNav() {
  const pathname = usePathname();
  const isCountryIntel = pathname.startsWith("/country/");

  return (
    <header className="h-[52px] flex items-center border-b border-gray-200 bg-white shrink-0">
      <div className="flex items-center justify-between w-full px-6">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <Globe className="h-6 w-6 text-primary" aria-hidden />
          <span className="font-semibold text-foreground">Sentinel AI</span>
        </div>

        {/* Tab links */}
        <nav className="flex items-center gap-1" aria-label="Main navigation">
          <Link
            href="/dashboard"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              pathname === "/dashboard"
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Dashboard
          </Link>
          <Link
            href="/globe"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              pathname === "/globe"
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Globe
          </Link>
          <Link
            href="/analytics"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              pathname === "/analytics"
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Analytics
          </Link>
          <Link
            href="/country/UA"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              isCountryIntel
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Country Intel
          </Link>
          <Link
            href="/threats"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              pathname === "/threats"
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Threat Feed
          </Link>
          <Link
            href="/reports"
            className={cn(
              "px-3 py-2 text-sm font-medium rounded-t transition-colors",
              pathname === "/reports"
                ? "text-primary border-b-2 border-primary bg-muted/50"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            Reports
          </Link>
        </nav>

        {/* Right: status, bell, ⌘K */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span
              className="h-2 w-2 rounded-full bg-green-500 animate-pulse"
              aria-hidden
            />
            <span>Systems Live</span>
          </div>
          <button
            type="button"
            className="relative p-2 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground"
            aria-label="Notifications (3)"
          >
            <Bell className="h-5 w-5" />
            <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-destructive text-[10px] font-medium text-destructive-foreground text-white">
              3
            </span>
          </button>
          <button
            type="button"
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground border rounded-md hover:bg-muted hover:text-foreground"
            aria-label="Search (⌘K)"
          >
            <Search className="h-4 w-4" />
            <span>⌘K</span>
          </button>
        </div>
      </div>
    </header>
  );
}
