import type { Transition, Variants } from "framer-motion";

export const spring = {
  default: {
    type: "spring",
    stiffness: 120,
    damping: 18,
    mass: 0.9,
  } as Transition,
  snappy: {
    type: "spring",
    stiffness: 300,
    damping: 24,
    mass: 0.6,
  } as Transition,
  slow: { type: "spring", stiffness: 60, damping: 20, mass: 1.2 } as Transition,
  bouncy: {
    type: "spring",
    stiffness: 200,
    damping: 12,
    mass: 0.8,
  } as Transition,
  gentle: {
    type: "spring",
    stiffness: 80,
    damping: 22,
    mass: 1.0,
  } as Transition,
};

export const cardInteraction = {
  whileHover: { y: -6, transition: spring.snappy },
  whileTap: { scale: 0.98, transition: spring.snappy },
};

export const focusCardInteraction = {
  whileHover: { y: -3, transition: spring.snappy },
  whileTap: { scale: 0.99, transition: spring.snappy },
};

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 32 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      ...spring.default,
      delay: i * 0.08,
    },
  }),
};

export const fadeLeft: Variants = {
  hidden: { opacity: 0, x: -24 },
  visible: (i: number = 0) => ({
    opacity: 1,
    x: 0,
    transition: { ...spring.default, delay: i * 0.05 },
  }),
};

export const scaleUp: Variants = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: (i: number = 0) => ({
    opacity: 1,
    scale: 1,
    transition: { ...spring.default, delay: i * 0.06 },
  }),
};

export const stagger: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.07 },
  },
};

export const pageTransition: Variants = {
  initial: { opacity: 0, y: 16 },
  animate: {
    opacity: 1,
    y: 0,
    transition: spring.default,
  },
  exit: {
    opacity: 0,
    y: -8,
    transition: { duration: 0.2, ease: "easeIn" },
  },
};

export const theme = {
  bg: "#070D12",
  surface: "#0B1216",
  card: "rgba(255,255,255,0.02)",
  cardBorder: "rgba(255,255,255,0.06)",
  cardHover: "rgba(255,255,255,0.04)",
  glow: "#00C896",
  glowDim: "rgba(0,200,150,0.08)",
  glowMid: "rgba(0,200,150,0.15)",
  secondary: "#2A7FFF",
  secondaryDim: "rgba(42,127,255,0.08)",
  text: "#E2E8F0",
  textMuted: "#8B95A5",
  textDim: "#4A5568",
  border: "rgba(255,255,255,0.06)",
};

export const surfaceCard =
  "bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.06)] rounded-xl backdrop-blur-sm";

export const interactiveCard =
  "bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.06)] rounded-xl backdrop-blur-sm transition-all duration-300 hover:border-[rgba(0,200,150,0.2)] hover:bg-[rgba(255,255,255,0.04)] hover:shadow-[0_0_30px_rgba(0,200,150,0.06)]";

export const focusCard =
  "bg-[rgba(255,255,255,0.03)] border border-[rgba(0,200,150,0.15)] rounded-xl backdrop-blur-md shadow-[0_0_40px_rgba(0,200,150,0.08)]";
