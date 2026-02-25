import { Sidebar } from "./Sidebar";

export function SidebarLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[calc(100vh-52px)] overflow-hidden">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-auto bg-muted/30">
        {children}
      </main>
    </div>
  );
}
