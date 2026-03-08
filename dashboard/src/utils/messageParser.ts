/**
 * Type-safe WebSocket message parser for FORGE server messages.
 *
 * Validates incoming JSON messages against expected schema before
 * dispatching to typed handlers.
 */

import type { SimulationState } from "../types/simulation";
import { createLogger } from "./logger";

const log = createLogger("messageParser");

/** Discriminated union of all server->client WebSocket messages. */
export type ServerMessage =
  | { type: "StateUpdate"; payload: SimulationState }
  | { type: "Error"; payload: string };

/**
 * Parse and validate a raw WebSocket message into a typed ServerMessage.
 *
 * @param data - Raw parsed JSON data from the WebSocket.
 * @returns A validated ServerMessage, or null if the message is malformed.
 */
export function parseServerMessage(data: unknown): ServerMessage | null {
  if (typeof data !== "object" || data === null) {
    log.warn("Expected object, got:", typeof data);
    return null;
  }

  const msg = data as Record<string, unknown>;

  if (typeof msg.type !== "string") {
    log.warn("Missing or invalid message type field");
    return null;
  }

  if (!("payload" in msg)) {
    log.warn("Missing payload field in message type:", msg.type);
    return null;
  }

  switch (msg.type) {
    case "StateUpdate": {
      if (typeof msg.payload !== "object" || msg.payload === null) {
        log.warn("Expected object payload for StateUpdate");
        return null;
      }
      const payload = msg.payload as Record<string, unknown>;
      if (typeof payload.tick !== "number" || !Array.isArray(payload.agents)) {
        log.warn("Invalid StateUpdate payload structure");
        return null;
      }
      return { type: "StateUpdate", payload: payload as unknown as SimulationState };
    }
    case "Error": {
      if (typeof msg.payload !== "string") {
        log.warn("Invalid Error payload — expected string");
        return null;
      }
      return { type: "Error", payload: msg.payload };
    }
    default:
      log.debug("Unknown message type:", msg.type);
      return null;
  }
}
