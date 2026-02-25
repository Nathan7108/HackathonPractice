"use client";

function formatTimestamp(d: Date): string {
  return d.toLocaleString("en-US", {
    month: "numeric",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

type PanelHeaderProps = {
  title: string;
  /** Optional; if not provided, client will show current time */
  timestamp?: string;
};

export function PanelHeader({ title, timestamp }: PanelHeaderProps) {
  const ts = timestamp ?? formatTimestamp(new Date());
  return (
    <div className="flex items-center justify-between px-3 py-2 border-b border-border">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
        {title}
      </h3>
      <span className="text-[10px] text-gray-400 tabular-nums">{ts}</span>
    </div>
  );
}
