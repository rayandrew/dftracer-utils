import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import { viteSingleFile } from "vite-plugin-singlefile";

// Emit a single self-contained dist/index.html (all JS/CSS inlined) that the
// C++ build embeds into the dftracer_server binary. During development, `npm
// run dev` proxies API calls to a locally running server on :8080.
export default defineConfig({
  plugins: [solid(), viteSingleFile()],
  server: {
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
  build: {
    target: "es2020",
    outDir: "dist",
    cssCodeSplit: false,
    assetsInlineLimit: 100000000,
    chunkSizeWarningLimit: 100000,
  },
});
