import { useEffect, useRef, type MouseEvent } from "react";
import { Boxes } from "lucide-react";
import type { AgentState, SimulationState } from "../types/simulation";
import { getConfig } from "../config/environment";
import {
  CANVAS_COLORS,
  findNearestAgent,
  healthFraction,
  HEALTH_HIGH_FRACTION,
  RENDER_GEOMETRY,
} from "../lib/canvasRender";
import {
  factionColor,
  INACTIVE_AGENT_COLOR,
  TERRAIN_COLORS,
} from "../lib/domainColors";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { EmptyState } from "./ui/empty-state";

interface SimulationCanvasProps {
  state: SimulationState | null;
  showGrid?: boolean;
  /** Currently selected agent id (highlighted). */
  selectedAgentId?: number | null;
  /** Invoked with the nearest agent when the canvas is clicked. */
  onSelectAgent?: (agent: AgentState) => void;
}

/** Canvas-based 2D renderer for the simulation grid. */
export function SimulationCanvas({
  state,
  showGrid,
  selectedAgentId,
  onSelectAgent,
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

    // Clear + base terrain (no per-tile data in SimulationState yet).
    ctx.fillStyle = CANVAS_COLORS.background;
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = TERRAIN_COLORS.Ground;
    ctx.fillRect(0, 0, width, height);

    if (gridLines) {
      ctx.strokeStyle = CANVAS_COLORS.gridLine;
      ctx.lineWidth = RENDER_GEOMETRY.gridLineWidth;
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

    for (const agent of state.agents) {
      drawAgent(ctx, agent, cellSize, agent.id === selectedAgentId);
    }
  }, [state, cellSize, gridLines, selectedAgentId]);

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !state || !onSelectAgent) return;
    const rect = canvas.getBoundingClientRect();
    // Map the click to canvas space, accounting for CSS scaling.
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = (event.clientX - rect.left) * scaleX;
    const py = (event.clientY - rect.top) * scaleY;

    const nearest = findNearestAgent(state.agents, { x: px, y: py }, cellSize);
    if (nearest) {
      onSelectAgent(nearest);
    }
  };

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <CardTitle>World</CardTitle>
        {state ? (
          <Badge variant="outline" className="font-mono">
            {state.gridWidth}×{state.gridHeight}
          </Badge>
        ) : null}
      </CardHeader>
      <CardContent className="flex flex-1 items-center justify-center overflow-auto bg-background/40">
        {state ? (
          // biome-ignore lint/a11y/useKeyWithClickEvents: agent selection is inherently spatial/pointer-based on the canvas; the agent inspector also reflects state non-visually.
          <canvas
            ref={canvasRef}
            onClick={handleClick}
            className="max-w-full rounded border border-border"
            style={{
              imageRendering: "pixelated",
              cursor: onSelectAgent ? "pointer" : "default",
            }}
          />
        ) : (
          <EmptyState
            icon={Boxes}
            title="Waiting for simulation"
            description="Connect a FORGE server to stream live world state into the canvas."
          />
        )}
      </CardContent>
    </Card>
  );
}

function drawAgent(
  ctx: CanvasRenderingContext2D,
  agent: AgentState,
  cellSize: number,
  selected: boolean,
) {
  const cx = agent.x * cellSize + cellSize / 2;
  const cy = agent.y * cellSize + cellSize / 2;
  const radius = cellSize * RENDER_GEOMETRY.agentRadiusFactor;

  if (selected) {
    ctx.strokeStyle = CANVAS_COLORS.selection;
    ctx.lineWidth = RENDER_GEOMETRY.selectionLineWidth;
    ctx.beginPath();
    ctx.arc(cx, cy, radius + RENDER_GEOMETRY.selectionRingPadding, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.fillStyle = agent.alive
    ? factionColor(agent.teamId ?? agent.id)
    : INACTIVE_AGENT_COLOR;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  if (agent.alive && agent.health > 0) {
    const barWidth = cellSize * RENDER_GEOMETRY.healthBarWidthFactor;
    const barHeight = RENDER_GEOMETRY.healthBarHeight;
    const barX = agent.x * cellSize + (cellSize - barWidth) / 2;
    const barY = agent.y * cellSize - RENDER_GEOMETRY.healthBarOffsetY;
    const healthFrac = healthFraction(agent.health);

    ctx.fillStyle = CANVAS_COLORS.healthTrack;
    ctx.fillRect(barX, barY, barWidth, barHeight);
    ctx.fillStyle =
      healthFrac > HEALTH_HIGH_FRACTION
        ? CANVAS_COLORS.healthHigh
        : CANVAS_COLORS.healthLow;
    ctx.fillRect(barX, barY, barWidth * healthFrac, barHeight);
  }

  if (agent.intent) {
    ctx.fillStyle = CANVAS_COLORS.intentLabel;
    const fontSize = Math.max(
      RENDER_GEOMETRY.minFontSize,
      cellSize * RENDER_GEOMETRY.fontSizeFactor,
    );
    ctx.font = `${fontSize}px monospace`;
    ctx.textAlign = "center";
    ctx.fillText(agent.intent, cx, cy + cellSize + 2);
  }
}
