import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Loader2, AlertCircle, Wifi, WifiOff, Send } from "lucide-react";

interface LiveInferenceProps {
  onDataReceived?: (data: any) => void;
}

export function LiveInference({ onDataReceived }: LiveInferenceProps) {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isServerOnline, setIsServerOnline] = useState<boolean | null>(null);
  const [lastResult, setLastResult] = useState<any>(null);

  // Check if backend is available
  useEffect(() => {
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  const checkServerStatus = async () => {
    try {
      const apiOrigin =
        (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
        "";
      const response = await fetch(`${apiOrigin}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(3000),
      });
      setIsServerOnline(response.ok);
    } catch {
      setIsServerOnline(false);
    }
  };

  const runInference = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const apiOrigin =
        (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
        "";
      const response = await fetch(`${apiOrigin}/api/inference/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Inference failed");
      }

      const data = await response.json();
      setLastResult(data);
      onDataReceived?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runInference();
    }
  };

  return (
    <div className="glass-card p-6">
      {/* Header with status indicator */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Zap size={20} className="text-bdh-accent" />
          Live Inference
        </h3>

        <div className="flex items-center gap-2">
          {isServerOnline === null ? (
            <span className="text-[#4A5568] text-sm flex items-center gap-1">
              <Loader2 size={14} className="animate-spin" />
              Checking...
            </span>
          ) : isServerOnline ? (
            <span className="text-green-400 text-sm flex items-center gap-1">
              <Wifi size={14} />
              Backend Online
            </span>
          ) : (
            <span className="text-orange-400 text-sm flex items-center gap-1">
              <WifiOff size={14} />
              Backend Offline
            </span>
          )}
        </div>
      </div>

      {/* Offline warning */}
      {isServerOnline === false && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg text-sm"
        >
          <p className="text-orange-300 flex items-center gap-2">
            <AlertCircle size={16} />
            Backend server not running. Start it with:
          </p>
          <code className="block mt-2 p-2 bg-[#0B1216] rounded text-xs text-[#CBD5E0]">
            python backend/live_server.py --model
            checkpoints/french/french_best.pt
          </code>
        </motion.div>
      )}

      {/* Input */}
      <div className="flex gap-3">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type any text to analyze (e.g., 'The capital of France is Paris')"
          className="input-field flex-1"
          disabled={!isServerOnline || isLoading}
        />
        <button
          onClick={runInference}
          disabled={!isServerOnline || isLoading || !inputText.trim()}
          className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <Send size={18} />
          )}
          Run
        </button>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-300 text-sm"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick stats from last result */}
      {lastResult && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 grid grid-cols-3 gap-4"
        >
          <div className="p-3 bg-white/[0.03] rounded-lg text-center">
            <div className="text-2xl font-bold text-bdh-accent">
              {(lastResult.overall_sparsity * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-[#8B95A5]">Sparsity</div>
          </div>
          <div className="p-3 bg-white/[0.03] rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-400">
              {lastResult.input_tokens.length}
            </div>
            <div className="text-xs text-[#8B95A5]">Tokens</div>
          </div>
          <div className="p-3 bg-white/[0.03] rounded-lg text-center">
            <div className="text-2xl font-bold text-green-400">
              {lastResult.frames.length}
            </div>
            <div className="text-xs text-[#8B95A5]">Frames</div>
          </div>
        </motion.div>
      )}

      {/* Example prompts */}
      <div className="mt-4">
        <p className="text-xs text-[#4A5568] mb-2">Try these examples:</p>
        <div className="flex flex-wrap gap-2">
          {[
            "The European Parliament",
            "<F:en>Hello world<T:fr>",
            "100 euros and 50 dollars",
            "France, Germany, and Spain",
          ].map((example) => (
            <button
              key={example}
              onClick={() => setInputText(example)}
              className="px-2 py-1 text-xs bg-white/5 hover:bg-white/10 rounded text-[#8B95A5] hover:text-[#E2E8F0] transition-colors"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
