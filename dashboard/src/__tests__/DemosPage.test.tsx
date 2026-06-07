import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DemosPage } from "../pages/DemosPage";

/** Build a fetch Response whose body streams the given SSE frames. */
function sseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return { ok: true, status: 200, body } as unknown as Response;
}

describe("DemosPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the section list and an idle output panel", () => {
    render(<DemosPage />);
    expect(screen.getByText("Demo Sections")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /World Generation/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("Select a demo to run")).toBeInTheDocument();
  });

  it("streams demo output and marks completion on the end marker", async () => {
    global.fetch = vi.fn().mockResolvedValue(
      sseResponse([
        `data: ${JSON.stringify("building world…")}\n\n`,
        `data: ${JSON.stringify("done generating")}\n\n`,
        `data: ${JSON.stringify("__STREAM_END__")}\n\n`,
      ]),
    );

    render(<DemosPage />);
    fireEvent.click(screen.getByRole("button", { name: /World Generation/ }));

    expect(await screen.findByText("building world…")).toBeInTheDocument();
    expect(screen.getByText("done generating")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("done")).toBeInTheDocument());
  });

  it("reports a non-OK backend response", async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue({ ok: false, status: 502, body: null } as Response);

    render(<DemosPage />);
    fireEvent.click(screen.getByRole("button", { name: /Navigation/ }));

    expect(
      await screen.findByText(/demo backend responded 502/),
    ).toBeInTheDocument();
  });

  it("reports a network failure with a hint", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("conn refused"));

    render(<DemosPage />);
    fireEvent.click(screen.getByRole("button", { name: /Crafting/ }));

    expect(await screen.findByText(/conn refused/)).toBeInTheDocument();
  });
});
