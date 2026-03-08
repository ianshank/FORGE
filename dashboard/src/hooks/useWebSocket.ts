import { useCallback, useEffect, useRef, useState } from "react";
import { getConfig } from "../config/environment";
import { createLogger } from "../utils/logger";

const log = createLogger("useWebSocket");

/** Default WebSocket reconnect interval in milliseconds. */
const WS_RECONNECT_DEFAULT_MS = 2000;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  reconnectInterval?: number;
}

/** Hook for managing a WebSocket connection with auto-reconnect. */
export function useWebSocket({
  url,
  onMessage,
  reconnectInterval,
}: UseWebSocketOptions) {
  const config = getConfig();
  const reconnectMs = reconnectInterval ?? WS_RECONNECT_DEFAULT_MS;
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onMessageRef = useRef(onMessage);
  const reconnectCount = useRef(0);

  // Keep onMessage ref current without causing reconnects
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (reconnectCount.current >= config.maxReconnectAttempts) {
      log.warn("Max reconnect attempts reached");
      return;
    }

    setStatus("connecting");
    log.info("Connecting to", url);

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      log.error(`Failed to create WebSocket with URL '${url}':`, err);
      setStatus("disconnected");
      reconnectCount.current += 1;
      if (reconnectCount.current < config.maxReconnectAttempts) {
        reconnectTimer.current = setTimeout(connect, reconnectMs);
      }
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      log.info("Connected");
      setStatus("connected");
      reconnectCount.current = 0;
    };
    ws.onclose = (event) => {
      log.warn(
        `Disconnected: code=${event.code}, reason='${event.reason}'`,
      );
      setStatus("disconnected");
      reconnectCount.current += 1;
      if (reconnectCount.current < config.maxReconnectAttempts) {
        reconnectTimer.current = setTimeout(connect, reconnectMs);
      }
    };
    ws.onerror = (event) => {
      log.error("WebSocket error:", event);
      ws.close();
    };
    ws.onmessage = (event) => {
      try {
        const data: unknown = JSON.parse(event.data as string);
        onMessageRef.current?.(data);
      } catch (err) {
        log.warn("Failed to parse message", err);
      }
    };
  }, [url, reconnectMs, config.maxReconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
    reconnectCount.current = 0;
    setStatus("disconnected");
    log.info("Disconnected (manual)");
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { status, disconnect, reconnect: connect };
}
