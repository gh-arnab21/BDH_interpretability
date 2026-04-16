import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  Zap,
  SkipForward,
  SkipBack,
  ChevronRight,
} from "lucide-react";
import { BDHArchitectureDiagram } from "@/features/architecture/BDHArchitectureDiagram";
import { MathDetailPanel } from "@/features/architecture/MathDetailPanel";
import { usePlaybackStore } from "@/stores/playbackStore";

const NUM_ARCH_STEPS = 13;
const STEP_DURATION = 2000;
const OUTPUT_DWELL = 500;

export function ArchitecturePage() {
  const [inputText, setInputText] = useState("The capital of France is Paris");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);

  const [currentTokenIdx, setCurrentTokenIdx] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [, setStepProgress] = useState(0);
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepStartRef = useRef<number>(0);

  const {
    playbackData,
    isLoading,
    loadingMessage,
    error,
    mode,
    loadPlayback,
    setFrame,
    reset,
  } = usePlaybackStore();

  const numTokens = playbackData?.input_chars?.length ?? 0;

  const findFrameIndex = useCallback(
    (tokenIdx: number, layer: number): number => {
      if (!playbackData) return 0;
      const idx = playbackData.frames.findIndex(
        (f) => f.token_idx === tokenIdx && f.layer === layer,
      );
      return idx >= 0 ? idx : 0;
    },
    [playbackData],
  );

  const currentFrameData = playbackData
    ? playbackData.frames[findFrameIndex(currentTokenIdx, currentLayer)]
    : undefined;

  useEffect(() => {
    if (!playbackData) return;
    const idx = findFrameIndex(currentTokenIdx, currentLayer);
    setFrame(idx);
  }, [currentTokenIdx, currentLayer, playbackData, findFrameIndex, setFrame]);

  useEffect(() => {
    loadPlayback(inputText);
  }, []);

  useEffect(() => {
    if (!isPlaying || !playbackData) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    stepStartRef.current = Date.now();
    setStepProgress(0);
    let dwelling = false;
    let dwellStart = 0;

    timerRef.current = setInterval(() => {
      // During dwell period, wait then advance
      if (dwelling) {
        if (Date.now() - dwellStart >= OUTPUT_DWELL) {
          dwelling = false;
          setCurrentTokenIdx((prevToken) => {
            const nextToken = prevToken + 1;
            if (nextToken >= numTokens) {
              setIsPlaying(false);
              return prevToken;
            }
            return nextToken;
          });
          setCurrentStep(0);
          stepStartRef.current = Date.now();
          setStepProgress(0);
        }
        return;
      }

      const elapsed = Date.now() - stepStartRef.current;
      const progress = Math.min(elapsed / STEP_DURATION, 1);
      setStepProgress(progress);

      if (progress >= 1) {
        // Step complete — advance
        setCurrentStep((prev) => {
          const nextStep = prev + 1;
          if (nextStep >= NUM_ARCH_STEPS) {
            // All steps done — enter dwell on output
            dwelling = true;
            dwellStart = Date.now();
            return prev; // stay on output step during dwell
          }
          stepStartRef.current = Date.now();
          setStepProgress(0);
          return nextStep;
        });
      }
    }, 30);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, playbackData, numTokens]);

  const handleRun = () => {
    setIsPlaying(false);
    setCurrentTokenIdx(0);
    setCurrentStep(0);
    setStepProgress(0);
    reset();
    loadPlayback(inputText);
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      // If we finished all tokens, restart
      if (
        currentTokenIdx >= numTokens - 1 &&
        currentStep >= NUM_ARCH_STEPS - 1
      ) {
        setCurrentTokenIdx(0);
        setCurrentStep(0);
      }
      setIsPlaying(true);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentTokenIdx(0);
    setCurrentStep(0);
    setStepProgress(0);
  };

  const handleNextToken = () => {
    setIsPlaying(false);
    if (currentTokenIdx < numTokens - 1) {
      setCurrentTokenIdx((p) => p + 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handlePrevToken = () => {
    setIsPlaying(false);
    if (currentTokenIdx > 0) {
      setCurrentTokenIdx((p) => p - 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handleNextStep = () => {
    setIsPlaying(false);
    if (currentStep < NUM_ARCH_STEPS - 1) {
      setCurrentStep((p) => p + 1);
      setStepProgress(1);
    } else if (currentTokenIdx < numTokens - 1) {
      setCurrentTokenIdx((p) => p + 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handleStepClick = (stepIdx: number) => {
    setIsPlaying(false);
    setCurrentStep(stepIdx);
    setStepProgress(1);
  };

  const handleTokenClick = (tokenIdx: number) => {
    setIsPlaying(false);
    setCurrentTokenIdx(tokenIdx);
    setCurrentStep(0);
    setStepProgress(0);
  };

  return (
    <div className="min-h-screen p-8" style={{ background: "#070D12" }}>
      {/* Loading overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm"
            style={{ background: "rgba(7,13,18,0.85)" }}
          >
            <div className="flex flex-col items-center gap-4 max-w-md text-center">
              <div className="relative w-16 h-16">
                <div
                  className="absolute inset-0 rounded-full border-4"
                  style={{ borderColor: "rgba(255,255,255,0.08)" }}
                />
                <div className="absolute inset-0 rounded-full border-4 border-t-[#00C896] border-r-transparent border-b-transparent border-l-transparent animate-spin" />
              </div>
              <p className="text-[#E2E8F0] text-sm font-medium">
                {loadingMessage || "Running inference on model..."}
              </p>
              <p className="text-[#6B7280] text-xs">
                The model processes each token through 6 layers with full
                activation extraction. This typically takes 10-30 seconds.
              </p>
              <LoadingTimer />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg"
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-red-300 font-medium mb-1">Inference Failed</p>
              <p className="text-red-400/80 text-sm">{error}</p>
            </div>
            <button
              onClick={handleRun}
              className="px-4 py-2 bg-red-500/30 hover:bg-red-500/50 text-red-200 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
            >
              Retry
            </button>
          </div>
        </motion.div>
      )}

      {/* Mode indicator */}
      {playbackData && mode === "live" && (
        <div className="mb-4 flex items-center gap-2">
          <span className="px-2 py-1 rounded text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/50">
            Live API
          </span>
          <span className="text-xs text-[#4A5568]">
            Real model inference — {playbackData.frames.length} frames extracted
          </span>
        </div>
      )}

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 120, damping: 18, mass: 0.9 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2 text-[#E2E8F0]">
          Interactive <span className="text-[#00C896]">Architecture</span>
        </h1>
        <p className="text-[#8B95A5]">
          Explore BDH's data flow with animated visualizations. Watch how ~95%
          of paths go dark at the ReLU — that's sparsity in action.
        </p>
      </motion.div>

      {/* Input Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6 mb-6"
      >
        <div className="flex gap-4">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to visualize..."
            className="input-field flex-1"
          />
          <button
            onClick={handleRun}
            disabled={isLoading}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                Running...
              </>
            ) : (
              <>
                <Zap size={18} className="mr-2" />
                Run
              </>
            )}
          </button>
        </div>

        {playbackData && (
          <div className="mt-4 flex items-center gap-4">
            {/* Token display - click to select */}
            <div className="flex-1 flex flex-wrap gap-1">
              {playbackData.input_chars.map((char, idx) => (
                <motion.span
                  key={idx}
                  onClick={() => handleTokenClick(idx)}
                  className={`px-2 py-1 rounded font-mono text-sm cursor-pointer transition-all hover:ring-2 hover:ring-[#00C896]/40 ${
                    currentTokenIdx === idx
                      ? "text-[#070D12] font-semibold"
                      : idx < currentTokenIdx
                        ? "text-[#8B95A5]"
                        : "text-[#4A5568]"
                  }`}
                  style={{
                    background:
                      currentTokenIdx === idx
                        ? "#00C896"
                        : idx < currentTokenIdx
                          ? "rgba(255,255,255,0.06)"
                          : "rgba(255,255,255,0.03)",
                    boxShadow:
                      currentTokenIdx === idx
                        ? "0 0 16px rgba(0,200,150,0.3)"
                        : "none",
                  }}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ delay: idx * 0.02 }}
                >
                  {char === " " ? "␣" : char}
                </motion.span>
              ))}
            </div>

            {/* Playback controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={handlePrevToken}
                disabled={currentTokenIdx === 0}
                className="p-2 rounded-lg transition-all duration-200 disabled:opacity-30 hover:scale-105"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
                title="Previous token"
              >
                <SkipBack size={18} />
              </button>
              <button
                onClick={handlePlayPause}
                className="p-2 rounded-lg transition-all duration-200 hover:scale-105"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
                title={isPlaying ? "Pause" : "Play (step-by-step)"}
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <button
                onClick={handleNextStep}
                className="p-2 rounded-lg transition-all duration-200 hover:scale-105"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
                title="Next step"
              >
                <ChevronRight size={20} />
              </button>
              <button
                onClick={handleNextToken}
                disabled={currentTokenIdx >= numTokens - 1}
                className="p-2 rounded-lg transition-all duration-200 disabled:opacity-30 hover:scale-105"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
                title="Next token"
              >
                <SkipForward size={18} />
              </button>
              <button
                onClick={handleReset}
                className="p-2 rounded-lg transition-all duration-200 hover:scale-105"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
                title="Reset"
              >
                <RotateCcw size={18} />
              </button>
            </div>
          </div>
        )}

        {/* Progress indicator */}
        {playbackData && (
          <div className="mt-3 flex items-center gap-3 text-xs text-[#6B7280]">
            <span>
              Token{" "}
              <span className="text-[#00C896] font-bold">
                {currentTokenIdx + 1}
              </span>
              /{numTokens}
            </span>
            <span className="text-[#2D3748]">|</span>
            <span>
              Step{" "}
              <span className="text-[#00C896] font-bold">
                {currentStep + 1}
              </span>
              /{NUM_ARCH_STEPS}
            </span>
            <span className="text-[#2D3748]">|</span>
            <span>
              Layer{" "}
              <span className="text-[#00C896] font-bold">
                {currentLayer + 1}
              </span>
            </span>
            <div className="flex-1" />
            {isPlaying && (
              <span className="text-green-400 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Playing
              </span>
            )}
          </div>
        )}
      </motion.div>

      {/* Main Diagram + Detail Panel */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-6 mb-6 overflow-x-auto"
      >
        <div className="flex gap-4">
          <div
            className={`flex-1 min-w-0 transition-all duration-300 ${selectedBlock !== null ? "" : ""}`}
          >
            <BDHArchitectureDiagram
              frameData={currentFrameData}
              playbackData={playbackData ?? undefined}
              currentLayer={currentLayer}
              isAnimating={isPlaying}
              currentStep={currentStep}
              onStepChange={handleStepClick}
              selectedBlock={selectedBlock}
              onBlockClick={(step) =>
                setSelectedBlock((prev) => (prev === step ? null : step))
              }
            />
          </div>
          {selectedBlock !== null && (
            <MathDetailPanel
              selectedBlock={selectedBlock}
              onClose={() => setSelectedBlock(null)}
              frameData={currentFrameData}
              playbackData={playbackData ?? undefined}
              currentLayer={currentLayer}
            />
          )}
        </div>
      </motion.div>

      {/* Layer selector */}
      {playbackData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Layer Selection</h3>
          <div className="flex gap-2 flex-wrap">
            {Array.from({ length: playbackData.num_layers }, (_, i) => (
              <button
                key={i}
                onClick={() => {
                  setCurrentLayer(i);
                  setIsPlaying(false);
                }}
                className={`px-4 py-2 rounded-lg font-mono transition-all duration-200 ${
                  currentLayer === i
                    ? "text-[#070D12] font-semibold"
                    : "text-[#6B7280]"
                }`}
                style={{
                  background:
                    currentLayer === i ? "#00C896" : "rgba(255,255,255,0.04)",
                  boxShadow:
                    currentLayer === i
                      ? "0 0 20px rgba(0,200,150,0.25)"
                      : "none",
                  border:
                    "1px solid " +
                    (currentLayer === i
                      ? "rgba(0,200,150,0.3)"
                      : "rgba(255,255,255,0.06)"),
                }}
              >
                L{i}
              </button>
            ))}
          </div>

          {/* Sparsity indicator */}
          {currentFrameData && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div
                className="p-4 rounded-lg"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
              >
                <div className="text-sm text-[#8B95A5] mb-1">X Sparsity</div>
                <div className="text-2xl font-bold text-[#00C896]">
                  {(currentFrameData.x_sparsity * 100).toFixed(1)}%
                </div>
                <div
                  className="mt-2 h-2 rounded-full overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.06)" }}
                >
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      background: "#00C896",
                      boxShadow: "0 0 8px rgba(0,200,150,0.4)",
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${currentFrameData.x_sparsity * 100}%` }}
                    transition={{ type: "spring", stiffness: 120, damping: 18 }}
                  />
                </div>
              </div>
              <div
                className="p-4 rounded-lg"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
              >
                <div className="text-sm text-[#8B95A5] mb-1">Y Sparsity</div>
                <div className="text-2xl font-bold text-[#2A7FFF]">
                  {(currentFrameData.y_sparsity * 100).toFixed(1)}%
                </div>
                <div
                  className="mt-2 h-2 rounded-full overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.06)" }}
                >
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      background: "#2A7FFF",
                      boxShadow: "0 0 8px rgba(42,127,255,0.4)",
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${currentFrameData.y_sparsity * 100}%` }}
                    transition={{ type: "spring", stiffness: 120, damping: 18 }}
                  />
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Key Insights Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-6 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-[#E2E8F0]">
          <Zap size={20} className="text-[#00C896]" />
          Key Insights
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <InsightCard
            title="Sparsification at ReLU"
            description="Watch the yellow encoder path expand from D=192 to N=3072 dimensions, then ReLU kills ~95% of activations. The diagram literally shows paths going dark."
          />
          <InsightCard
            title="Linear Attention"
            description="The blue attention block computes ϱ += x^T v — a rank-1 update that's O(T) not O(T²). This is why BDH scales to unlimited context."
          />
          <InsightCard
            title="Hebbian State"
            description="The attention state ϱ accumulates co-activation patterns. This IS the Hebbian memory — neurons that fire together strengthen their connection."
          />
          <InsightCard
            title="Gating (x × y)"
            description="The element-wise multiplication of sparse x and y creates even sparser output. Only paths active in BOTH survive — biological-like signal gating."
          />
        </div>
      </motion.div>
    </div>
  );
}

function InsightCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div
      className="p-4 rounded-lg transition-all duration-300 hover:translate-y-[-2px]"
      style={{
        background: "rgba(255,255,255,0.02)",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <h4 className="font-medium text-[#00C896] mb-2">{title}</h4>
      <p className="text-sm text-[#8B95A5]">{description}</p>
    </div>
  );
}

function LoadingTimer() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - start) / 1000)),
      500,
    );
    return () => clearInterval(id);
  }, []);
  return (
    <p className="text-[#4A5568] text-xs font-mono">Elapsed: {elapsed}s</p>
  );
}
