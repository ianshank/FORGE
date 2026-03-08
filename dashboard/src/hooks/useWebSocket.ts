import { useCallback, useEffect, useRef, useState } from "react";

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

const MAX_RECONNECT_ATTEMPTS = 10;

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  reconnectInterval?: number;
}

/** Hook for managing a WebSocket connection with auto-reconnect. */
export function useWebSocket({
  url,
  onMessage,
  reconnectInterval = 3000,
}: UseWebSocketOptions) {
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
    if (reconnectCount.current >= MAX_RECONNECT_ATTEMPTS) return;

    setStatus("connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.error(`Failed to create WebSocket with URL '${url}':`, err);
      setStatus("disconnected");
      reconnectCount.current += 1;
      if (reconnectCount.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      }
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      reconnectCount.current = 0;
    };
    ws.onclose = (event) => {
      console.warn(
        `WebSocket disconnected: code=${event.code}, reason='${event.reason}'`,
      );
      setStatus("disconnected");
      reconnectCount.current += 1;
      if (reconnectCount.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      }
    };
    ws.onerror = (event) => {
      console.error("WebSocket error:", event);
      ws.close();
    };
    ws.onmessage = (event) => {
      try {
        const data: unknown = JSON.parse(event.data as string);
        onMessageRef.current?.(data);
      } catch (err) {
        console.warn("WebSocket: failed to parse message", err);
      }
    };
  }, [url, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
    reconnectCount.current = 0;
    setStatus("disconnected");
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { status, disconnect, reconnect: connect };
}
