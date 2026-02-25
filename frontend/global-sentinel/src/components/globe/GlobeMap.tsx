"use client";

import { useRef, useEffect } from "react";
import "mapbox-gl/dist/mapbox-gl.css";

const COUNTRY_CENTERS: Record<string, [number, number]> = {
  UA: [31.16, 48.38],
  IR: [53.69, 32.43],
  PK: [69.35, 30.38],
  ET: [40.49, 9.15],
  VE: [-66.59, 6.42],
  TW: [120.96, 23.69],
  RS: [21.01, 44.02],
  BR: [-51.93, -14.24],
};

const RISK_FILL: Record<string, string> = {
  UA: "#dc2626",
  IR: "#dc2626",
  PK: "#ea580c",
  ET: "#ea580c",
  VE: "#eab308",
  TW: "#eab308",
  RS: "#22c55e",
  BR: "#22c55e",
};

const GEOJSON_URL =
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson";

type Props = {
  onCountrySelect: (code: string) => void;
};

export function GlobeMap({ onCountrySelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);

  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token || !containerRef.current) return;

    async function initMap() {
      const mapboxgl = (await import("mapbox-gl")).default;

      const map = new mapboxgl.Map({
        container: containerRef.current!,
        style: "mapbox://styles/mapbox/light-v11",
        center: [20, 20],
        zoom: 1.5,
        projection: "globe",
      });

      map.on("load", () => {
        map.addSource("countries", {
          type: "geojson",
          data: GEOJSON_URL,
        });
        map.addLayer({
          id: "country-fill",
          type: "fill",
          source: "countries",
          paint: {
            "fill-color": [
              "match",
              ["get", "ISO_A2"],
              "UA",
              RISK_FILL.UA,
              "IR",
              RISK_FILL.IR,
              "PK",
              RISK_FILL.PK,
              "ET",
              RISK_FILL.ET,
              "VE",
              RISK_FILL.VE,
              "TW",
              RISK_FILL.TW,
              "RS",
              RISK_FILL.RS,
              "BR",
              RISK_FILL.BR,
              "#e2e5ea",
            ],
            "fill-opacity": 0.85,
          },
        });
        map.addLayer({
          id: "country-outline",
          type: "line",
          source: "countries",
          paint: {
            "line-color": "#94a3b8",
            "line-width": 0.5,
          },
        });

        map.on("click", "country-fill", (e) => {
          const feat = e.features?.[0];
          const code = feat?.properties?.ISO_A2;
          if (code && COUNTRY_CENTERS[code]) {
            onCountrySelect(code);
            map.flyTo({
              center: COUNTRY_CENTERS[code],
              zoom: 3,
              duration: 1000,
            });
          }
        });
        map.getCanvas().style.cursor = "pointer";
      });

      mapRef.current = map;
      return () => {
        map.remove();
        mapRef.current = null;
      };
    }

    const cleanup = initMap();
    return () => {
      cleanup.then((fn) => fn?.());
    };
  }, [onCountrySelect]);

  return <div ref={containerRef} className="w-full h-full min-h-[400px]" />;
}
