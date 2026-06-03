import { Activity } from "lucide-react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { AreaTrend } from "../components/ui/area-trend";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { EmptyState } from "../components/ui/empty-state";
import { Input } from "../components/ui/input";
import { Skeleton } from "../components/ui/skeleton";
import { Slider } from "../components/ui/slider";
import { StatCard } from "../components/ui/stat-card";
import { StatusDot } from "../components/ui/status-dot";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";

describe("Button", () => {
  it("renders and handles clicks", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Go</Button>);
    fireEvent.click(screen.getByRole("button", { name: "Go" }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("supports variants and sizes", () => {
    render(
      <Button variant="danger" size="icon" aria-label="del">
        x
      </Button>,
    );
    expect(screen.getByRole("button", { name: "del" })).toBeInTheDocument();
  });

  it("renders as a child element when asChild is set", () => {
    render(
      <Button asChild>
        <a href="/x">link</a>
      </Button>,
    );
    expect(screen.getByRole("link", { name: "link" })).toHaveAttribute("href", "/x");
  });

  it("forwards a ref", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>r</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});

describe("Badge", () => {
  it("renders each variant", () => {
    for (const variant of ["default", "primary", "success", "warning", "danger", "outline"] as const) {
      const { unmount } = render(<Badge variant={variant}>{variant}</Badge>);
      expect(screen.getByText(variant)).toBeInTheDocument();
      unmount();
    }
  });
});

describe("Card", () => {
  it("composes header, title and content", () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Title</CardTitle>
        </CardHeader>
        <CardContent>Body</CardContent>
      </Card>,
    );
    expect(screen.getByText("Title")).toBeInTheDocument();
    expect(screen.getByText("Body")).toBeInTheDocument();
  });
});

describe("Input", () => {
  it("accepts user input", () => {
    const onChange = vi.fn();
    render(<Input aria-label="field" onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("field"), {
      target: { value: "abc" },
    });
    expect(onChange).toHaveBeenCalled();
  });
});

describe("Skeleton", () => {
  it("renders a placeholder block", () => {
    const { container } = render(<Skeleton className="h-4" />);
    expect(container.firstChild).toHaveClass("animate-pulse");
  });
});

describe("Slider", () => {
  it("renders a slider role", () => {
    render(<Slider value={[10]} min={0} max={100} aria-label="vol" />);
    expect(screen.getByRole("slider")).toBeInTheDocument();
  });
});

describe("StatCard", () => {
  it("renders label, value, unit and icon", () => {
    render(
      <StatCard label="Tick" value="42" unit="ms" icon={Activity} tone="primary" />,
    );
    expect(screen.getByText("Tick")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("ms")).toBeInTheDocument();
  });

  it("renders without an icon or unit", () => {
    render(<StatCard label="Plain" value="0" />);
    expect(screen.getByText("Plain")).toBeInTheDocument();
  });
});

describe("StatusDot", () => {
  it("labels each connection status", () => {
    for (const status of ["connected", "connecting", "disconnected"] as const) {
      const { unmount } = render(<StatusDot status={status} />);
      expect(screen.getByRole("img")).toBeInTheDocument();
      unmount();
    }
  });
});

describe("EmptyState", () => {
  it("renders title, description, icon and action", () => {
    render(
      <EmptyState
        icon={Activity}
        title="Nothing"
        description="No data"
        action={<button type="button">Act</button>}
      />,
    );
    expect(screen.getByText("Nothing")).toBeInTheDocument();
    expect(screen.getByText("No data")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Act" })).toBeInTheDocument();
  });

  it("renders without optional props", () => {
    render(<EmptyState title="Bare" />);
    expect(screen.getByText("Bare")).toBeInTheDocument();
  });
});

describe("Tabs", () => {
  it("marks the active tab and switches selection on click", () => {
    render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">A</TabsTrigger>
          <TabsTrigger value="b">B</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Panel A</TabsContent>
        <TabsContent value="b">Panel B</TabsContent>
      </Tabs>,
    );
    const tabA = screen.getByRole("tab", { name: "A" });
    const tabB = screen.getByRole("tab", { name: "B" });
    expect(tabA).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Panel A")).toBeInTheDocument();

    fireEvent.mouseDown(tabB);
    fireEvent.click(tabB);
    expect(tabB).toHaveAttribute("aria-selected", "true");
  });
});

describe("AreaTrend", () => {
  const data = [
    { index: 0, value: 1 },
    { index: 1, value: 2 },
    { index: 2, value: 3 },
  ];

  it("renders without a reference line", () => {
    const { container } = render(
      <AreaTrend data={data} xKey="index" dataKey="value" color="#39ff7e" />,
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });

  it("renders with a reference line and no grid", () => {
    const { container } = render(
      <AreaTrend
        data={data}
        xKey="index"
        dataKey="value"
        color="#39ff7e"
        referenceX={1}
        showGrid={false}
        height={120}
      />,
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeTruthy();
  });
});
