import { useEffect, useRef } from "react";
import type { AgentState, SimulationState } from "../types/simulation";
import { getConfig } from "../config/environment";

/** Terrain type → hex color mapping. */
const TERRAIN_COLORS: Record<string, string> = {
  Ground: "#8B9556",
  Water: "#4A90D9",
  Wall: "#6B6B6B",
  Lava: "#D94A4A",
  Ice: "#B0D4E8",
  Sand: "#D4C07A",
  Forest: "#2D6B3F",
  Mountain: "#8B7355",
};

/** Faction → color mapping. */
const FACTION_COLORS: string[] = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"];

interface SimulationCanvasProps {
  state: SimulationState | null;
  showGrid?: boolean;
}

/** Canvas-based 2D renderer for the simulation grid. */
export function SimulationCanvas({
  state,
  showGrid,
}: SimulationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const config = getConfig();
  const cellSize = config.cellSize;
  const gridLines = showGrid ?? config.showGridLines;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = state.gridWidth * cellSize;
    const height = state.gridHeight * cellSize;
    canvas.width = width;
    canvas.height = height;

    // Clear
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    // Draw terrain (simplified — no per-tile data in SimulationState yet)
    ctx.fillStyle = TERRAIN_COLORS.Ground;
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    if (gridLines) {
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.lineWidth = 0.5;
      for (let x = 0; x <= state.gridWidth; x++) {
        ctx.beginPath();
        ctx.moveTo(x * cellSize, 0);
        ctx.lineTo(x * cellSize, height);
        ctx.stroke();
      }
      for (let y = 0; y <= state.gridHeight; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * cellSize);
        ctx.lineTo(width, y * cellSize);
        ctx.stroke();
      }
    }

    // Draw agents
    for (const agent of state.agents) {
      drawAgent(ctx, agent, cellSize);
    }
  }, [state, cellSize, gridLines]);

  return (
    <canvas
      ref={canvasRef}
      className="border border-gray-700 rounded"
      style={{ imageRendering: "pixelated" }}
    />
  );
}

function drawAgent(
  ctx: CanvasRenderingContext2D,
  agent: AgentState,
  cellSize: number,
) {
  const cx = agent.x * cellSize + cellSize / 2;
  const cy = agent.y * cellSize + cellSize / 2;
  const radius = cellSize * 0.35;

  // Agent circle
  const colorIdx = (agent.teamId ?? agent.id) % FACTION_COLORS.length;
  ctx.fillStyle = agent.alive ? FACTION_COLORS[colorIdx] : "#666";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  // Health bar
  if (agent.alive && agent.health > 0) {
    const barWidth = cellSize * 0.8;
    const barHeight = 2;
    const barX = agent.x * cellSize + (cellSize - barWidth) / 2;
    const barY = agent.y * cellSize - 3;
    const healthFrac = Math.min(agent.health / 655360, 1); // 10.0 fixed-point

    ctx.fillStyle = "#333";
    ctx.fillRect(barX, barY, barWidth, barHeight);
    ctx.fillStyle = healthFrac > 0.5 ? "#22c55e" : "#ef4444";
    ctx.fillRect(barX, barY, barWidth * healthFrac, barHeight);
  }

  // Intent label
  if (agent.intent) {
    ctx.fillStyle = "#fff";
    ctx.font = `${Math.max(8, cellSize * 0.6)}px monospace`;
    ctx.textAlign = "center";
    ctx.fillText(agent.intent, cx, cy + cellSize + 2);
  }
}
