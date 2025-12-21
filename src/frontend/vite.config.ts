import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const pollingRaw = String(process.env.CHOKIDAR_USEPOLLING || "").toLowerCase();
const usePolling = pollingRaw ? !["0", "false", "no"].includes(pollingRaw) : true;
const interval = Number(process.env.CHOKIDAR_INTERVAL || "500") || 500;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      usePolling,
      interval,
      ignored: [
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
        "**/checkpoints/**",
        "**/rollouts/**",
        "**/temp/**",
        "**/gameplay_videos/**",
        "**/highlights/**",
        "**/game_states/**",
      ],
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8040",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
