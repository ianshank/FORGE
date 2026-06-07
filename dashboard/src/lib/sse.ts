/**
 * Minimal parser for Server-Sent Events delivered over a `fetch` body stream.
 *
 * The demo backend emits frames as `data: <json-encoded-line>\n\n`. This module
 * isolates the (otherwise fiddly) framing logic so it can be unit-tested
 * without a network, and reused by any SSE consumer.
 */

/** Line prefix carrying the event payload. */
export const SSE_DATA_PREFIX = "data:";

/** Payloads that signal the producer has finished the stream. */
export const STREAM_END_MARKERS = ["__STREAM_END__", "__DONE__"] as const;

/** Outcome of parsing a (possibly partial) buffer of SSE text. */
export interface SseParseResult {
  /** Fully-received event payloads, in order. */
  events: string[];
  /** Trailing partial frame to carry into the next read. */
  rest: string;
}

/**
 * Split a buffer into complete SSE events plus any trailing partial frame.
 *
 * Each event's `data:` payload is JSON-decoded when possible (the backend
 * JSON-encodes each line); otherwise the trimmed raw text is used.
 */
export function parseSseBuffer(buffer: string): SseParseResult {
  const frames = buffer.split("\n\n");
  // The final element is an incomplete frame (no terminating blank line yet).
  const rest = frames.pop() ?? "";
  const events: string[] = [];

  for (const frame of frames) {
    const dataLine = frame
      .split("\n")
      .find((line) => line.startsWith(SSE_DATA_PREFIX));
    if (!dataLine) continue;

    const raw = dataLine.slice(SSE_DATA_PREFIX.length).trim();
    let payload: string;
    try {
      const decoded: unknown = JSON.parse(raw);
      payload = typeof decoded === "string" ? decoded : raw;
    } catch {
      payload = raw;
    }
    events.push(payload);
  }

  return { events, rest };
}

/** Whether a payload marks the end of the stream. */
export function isStreamEnd(payload: string): boolean {
  return (STREAM_END_MARKERS as readonly string[]).includes(payload);
}
