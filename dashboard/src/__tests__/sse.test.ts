import { describe, expect, it } from "vitest";
import {
  isStreamEnd,
  parseSseBuffer,
  STREAM_END_MARKERS,
} from "../lib/sse";

describe("parseSseBuffer", () => {
  it("parses a single complete JSON-encoded frame", () => {
    const { events, rest } = parseSseBuffer(`data: ${JSON.stringify("hello")}\n\n`);
    expect(events).toEqual(["hello"]);
    expect(rest).toBe("");
  });

  it("parses multiple frames in one buffer", () => {
    const buffer = `data: ${JSON.stringify("a")}\n\ndata: ${JSON.stringify("b")}\n\n`;
    expect(parseSseBuffer(buffer).events).toEqual(["a", "b"]);
  });

  it("carries a trailing partial frame in `rest`", () => {
    const buffer = `data: ${JSON.stringify("done")}\n\ndata: partial`;
    const { events, rest } = parseSseBuffer(buffer);
    expect(events).toEqual(["done"]);
    expect(rest).toBe("data: partial");
  });

  it("falls back to raw text when payload is not valid JSON", () => {
    const { events } = parseSseBuffer("data: not-json\n\n");
    expect(events).toEqual(["not-json"]);
  });

  it("keeps raw text when JSON decodes to a non-string", () => {
    const { events } = parseSseBuffer("data: 42\n\n");
    expect(events).toEqual(["42"]);
  });

  it("ignores frames without a data line", () => {
    const { events } = parseSseBuffer("event: ping\n\n");
    expect(events).toEqual([]);
  });

  it("trims the data prefix and surrounding whitespace", () => {
    const { events } = parseSseBuffer(`data:   ${JSON.stringify("x")}  \n\n`);
    expect(events).toEqual(["x"]);
  });
});

describe("isStreamEnd", () => {
  it("recognises every end marker", () => {
    for (const marker of STREAM_END_MARKERS) {
      expect(isStreamEnd(marker)).toBe(true);
    }
  });

  it("returns false for ordinary payloads", () => {
    expect(isStreamEnd("regular line")).toBe(false);
  });
});
