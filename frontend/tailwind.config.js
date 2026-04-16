/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        /* Neural Observatory palette */
        obs: {
          bg: "#070D12",
          surface: "#0B1216",
          card: "rgba(255,255,255,0.02)",
          "card-hover": "rgba(255,255,255,0.04)",
          glow: "#00C896",
          "glow-dim": "rgba(0,200,150,0.08)",
          secondary: "#2A7FFF",
          "secondary-dim": "rgba(42,127,255,0.08)",
          border: "rgba(255,255,255,0.06)",
        },
        /* Legacy BDH brand (kept for data viz) */
        bdh: {
          dark: "#070D12",
          darker: "#050A0E",
          accent: "#00C896",
          "accent-light": "#34D399",
          encoder: "#fef3c7",
          "encoder-stroke": "#d97706",
          attention: "#d1fae5",
          "attention-stroke": "#059669",
          relu: "#fee2e2",
          "relu-stroke": "#dc2626",
          french: "#3b82f6",
          portuguese: "#10b981",
          merged: "#8b5cf6",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        flow: "flow 2s ease-in-out infinite",
        "glow-pulse": "glowPulse 3s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "dash-flow": "dashFlow 2s linear infinite",
      },
      keyframes: {
        flow: {
          "0%, 100%": { transform: "translateY(0)", opacity: "1" },
          "50%": { transform: "translateY(-10px)", opacity: "0.8" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 20px rgba(0,200,150,0.08)" },
          "50%": { boxShadow: "0 0 40px rgba(0,200,150,0.18)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
        dashFlow: {
          to: { strokeDashoffset: "-100" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "noise": `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E")`,
        "grid-pattern": `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
      },
    },
  },
  plugins: [],
};
