import { BrowserRouter, Routes, Route } from "react-router-dom";
import React, { lazy, Suspense } from "react";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { ArchitecturePage } from "./pages/ArchitecturePage";
import { SparsityPage } from "./pages/SparsityPage";
import { MonosemanticityPage } from "./pages/MonosemanticityPage";
import { HebbianPage } from "./pages/HebbianPage";
import { MergePage } from "./pages/MergePage";
import { FindingsPage } from "./pages/FindingsPage";
import { LearnBDHPage } from "./pages/LearnBDHPage";

const GraphPage = lazy(() =>
  import("./pages/GraphPage").then((m) => ({ default: m.GraphPage })),
);
const GraphTest = lazy(() =>
  import("./pages/GraphTest").then((m) => ({ default: m.GraphTest })),
);

function Loading() {
  return (
    <div className="flex items-center justify-center h-full text-[#4A5568]">
      Loading 3D view…
    </div>
  );
}

class RouteErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: "" };
  }
  static getDerivedStateFromError(err: Error) {
    return { hasError: true, error: err.message };
  }
  componentDidCatch(err: Error) {
    console.error("[RouteErrorBoundary]", err);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
          <div className="text-yellow-500 text-4xl">⚠️</div>
          <h2 className="text-lg font-semibold text-[#E2E8F0]">
            This page encountered an error
          </h2>
          <p className="text-sm text-[#8B95A5] text-center max-w-md">
            {this.state.error}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: "" })}
            className="px-4 py-2 bg-[#00C896]/10 border border-[#00C896]/20 text-[#00C896] text-sm rounded-lg hover:bg-[#00C896]/20 transition-colors"
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Diagnostic route — remove after testing */}
        <Route
          path="/graph-test"
          element={
            <Suspense fallback={<Loading />}>
              <GraphTest />
            </Suspense>
          }
        />
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="architecture" element={<ArchitecturePage />} />
          <Route path="sparsity" element={<SparsityPage />} />
          <Route
            path="graph"
            element={
              <RouteErrorBoundary>
                <Suspense fallback={<Loading />}>
                  <GraphPage />
                </Suspense>
              </RouteErrorBoundary>
            }
          />
          <Route path="monosemanticity" element={<MonosemanticityPage />} />
          <Route path="hebbian" element={<HebbianPage />} />
          <Route path="merge" element={<MergePage />} />
          <Route
            path="findings"
            element={
              <RouteErrorBoundary>
                <FindingsPage />
              </RouteErrorBoundary>
            }
          />
          <Route path="learn" element={<LearnBDHPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
