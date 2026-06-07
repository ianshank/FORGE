import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { App } from "../App";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App routing", () => {
  it("redirects the index route to the live view", async () => {
    renderAt("/");
    expect(await screen.findByText("Waiting for simulation")).toBeInTheDocument();
    // Shell chrome is present.
    expect(screen.getByText("FORGE")).toBeInTheDocument();
  });

  it("renders the runs route", async () => {
    renderAt("/runs");
    expect(await screen.findByText("No runs recorded")).toBeInTheDocument();
  });

  it("renders the settings route", async () => {
    renderAt("/settings");
    expect(
      await screen.findByText("Runtime Configuration"),
    ).toBeInTheDocument();
  });

  it("redirects unknown routes to the live view", async () => {
    renderAt("/does-not-exist");
    expect(await screen.findByText("Waiting for simulation")).toBeInTheDocument();
  });
});
