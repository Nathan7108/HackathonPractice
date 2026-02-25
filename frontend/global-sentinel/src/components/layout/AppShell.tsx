"use client";

import { useState } from "react";
import Link from "next/link";
import { Globe, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TopNav } from "@/components/layout/TopNav";
import { Sidebar } from "@/components/layout/Sidebar";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-[#F6F9FB]">
      {/* Single seamless header: grey bar always visible at top */}
      <header className="h-[52px] flex-shrink-0 flex items-center bg-[#F6F9FB] w-full">
        {/* Left: logo (fixed) + name + collapse — same width as sidebar so they align */}
        <div
          className={cn(
            "relative z-10 h-full flex items-center shrink-0 rounded-tl-lg transition-all duration-200",
            isCollapsed ? "w-[60px] justify-center" : "w-[260px]"
          )}
          aria-label="App sidebar"
          aria-expanded={!isCollapsed}
        >
          {!isCollapsed && (
            <Link
              href="/dashboard"
              className="h-full flex items-center justify-center shrink-0 w-14"
              aria-label="Sentinel AI home"
            >
              <Globe className="text-primary h-6 w-6" aria-hidden />
            </Link>
          )}
          <div
            className={cn(
              "flex items-center h-full",
              isCollapsed ? "absolute inset-0 justify-center" : "flex-1 min-w-0 justify-between px-2"
            )}
          >
            {!isCollapsed && (
              <span className="font-semibold text-slate-800 truncate">Sentinel AI</span>
            )}
            <Button
              variant="ghost"
              size="icon"
              type="button"
              className="relative z-10 h-9 w-9 shrink-0 text-slate-600 hover:bg-slate-200/60 hover:text-slate-900 pointer-events-auto"
              onClick={() => setIsCollapsed((c) => !c)}
              aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {isCollapsed ? (
                <ChevronRight className="h-5 w-5" />
              ) : (
                <ChevronLeft className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>
        {/* Right: nav tabs + actions */}
        <TopNav />
      </header>

      {/* Body: sidebar (grey left) + white shell; only main content scrolls */}
      <div className="flex flex-1 min-h-0 overflow-hidden pt-2 pr-2 pb-1">
        <aside
          className={cn(
            "z-10 flex-shrink-0 flex flex-col bg-[#F6F9FB] rounded-tl-lg overflow-hidden self-stretch transition-all duration-200",
            isCollapsed ? "w-[60px]" : "w-[260px]"
          )}
        >
          <Sidebar
            isCollapsed={isCollapsed}
            setIsCollapsed={setIsCollapsed}
            skipToggle
          />
        </aside>
        <main className="flex-1 min-w-0 min-h-0 flex flex-col rounded-lg overflow-hidden bg-white shadow-sm">
          <div className="flex-1 min-h-0 overflow-auto">
            {children}
          </div>
        </main>
      </div>

      {/* Footer: grey bar always visible at bottom */}
      <footer className="h-8 flex-shrink-0 w-full bg-[#F6F9FB]" aria-hidden />
    </div>
  );
}
