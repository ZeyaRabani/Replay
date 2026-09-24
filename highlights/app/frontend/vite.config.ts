import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// MOCK=1 npm run dev -> proxy /api to the mock server (npm run mock) instead of the FastAPI backend
const target = process.env.MOCK === "1" ? "http://127.0.0.1:8001" : (process.env.BACKEND_URL ?? "http://127.0.0.1:8000");

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": target,
    },
  },
});
