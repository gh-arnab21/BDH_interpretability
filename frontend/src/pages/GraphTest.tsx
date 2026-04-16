/**
 * Minimal diagnostic test for ForceGraph3D.
 * If this renders colored spheres → the library works and the issue is in GraphPage.
 * If this is blank too → the library/build is broken.
 */
import { useRef, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";

const HARDCODED = {
  nodes: [
    { id: 1, color: "#ff0000", val: 10 },
    { id: 2, color: "#00ff00", val: 10 },
    { id: 3, color: "#0000ff", val: 10 },
    { id: 4, color: "#ffff00", val: 10 },
    { id: 5, color: "#ff00ff", val: 10 },
    { id: 6, color: "#00ffff", val: 8 },
    { id: 7, color: "#ff8800", val: 8 },
    { id: 8, color: "#8800ff", val: 8 },
  ],
  links: [
    { source: 1, target: 2 },
    { source: 1, target: 3 },
    { source: 2, target: 4 },
    { source: 3, target: 5 },
    { source: 4, target: 5 },
    { source: 5, target: 6 },
    { source: 6, target: 7 },
    { source: 7, target: 8 },
    { source: 8, target: 1 },
  ],
};

export function GraphTest() {
  const fgRef = useRef<any>(null);

  useEffect(() => {
    console.log("[TEST] GraphTest mounted. ForceGraph3D ref:", fgRef.current);
    // Force camera close
    setTimeout(() => {
      if (fgRef.current) {
        console.log("[TEST] Fitting camera...");
        fgRef.current.zoomToFit(300, 50);
      }
    }, 2000);
  }, []);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#111" }}>
      <div
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          zIndex: 999,
          color: "white",
          fontSize: 14,
          background: "rgba(0,0,0,0.7)",
          padding: "8px 12px",
          borderRadius: 6,
        }}
      >
        ForceGraph3D Diagnostic Test — 8 nodes, 9 links
      </div>
      <ForceGraph3D
        ref={fgRef}
        graphData={HARDCODED}
        nodeColor="color"
        nodeVal="val"
        nodeRelSize={8}
        linkColor={() => "#ffffff"}
        linkOpacity={0.6}
        linkWidth={1}
        backgroundColor="#111111"
        width={window.innerWidth}
        height={window.innerHeight}
        warmupTicks={50}
        cooldownTicks={100}
      />
    </div>
  );
}
