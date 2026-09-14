import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages sub-path (https://<user>.github.io/Projects/).
// Overridable so `npm run dev` and local previews serve from root.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE ?? "/Projects/",
  build: { outDir: "dist", assetsInlineLimit: 0, chunkSizeWarningLimit: 1200 },
});
