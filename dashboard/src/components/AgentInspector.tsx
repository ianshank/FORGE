import { MousePointerClick } from "lucide-react";
import type { AgentState } from "../types/simulation";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { EmptyState } from "./ui/empty-state";

interface AgentInspectorProps {
  agent: AgentState | null;
}

interface FieldProps {
  label: string;
  value: string;
  mono?: boolean;
  accent?: boolean;
}

function Field({ label, value, mono, accent }: FieldProps) {
  return (
    <div className="flex items-center justify-between border-b border-border/60 py-1.5 last:border-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd
        className={`text-xs ${mono ? "font-mono" : ""} ${
          accent ? "text-primary" : "text-foreground"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

/** Detailed inspector panel for a single agent. */
export function AgentInspector({ agent }: AgentInspectorProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent Inspector</CardTitle>
        {agent ? (
          <Badge variant={agent.alive ? "success" : "danger"}>
            {agent.alive ? "Alive" : "Dead"}
          </Badge>
        ) : null}
      </CardHeader>
      <CardContent>
        {agent ? (
          <dl>
            <Field label="ID" value={`#${agent.id}`} mono />
            <Field
              label="Position"
              value={`(${agent.x}, ${agent.y})`}
              mono
            />
            <Field label="Health" value={String(agent.health)} mono />
            <Field label="Team" value={agent.teamId?.toString() ?? "None"} />
            <Field label="Vision" value={String(agent.visionRadius)} mono />
            {agent.intent ? (
              <Field label="Intent" value={agent.intent} accent />
            ) : null}
          </dl>
        ) : (
          <EmptyState
            icon={MousePointerClick}
            title="No agent selected"
            description="Click an agent in the world view to inspect its state."
          />
        )}
      </CardContent>
    </Card>
  );
}
