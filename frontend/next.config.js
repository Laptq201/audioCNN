/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  eslint: {
    ignoreDuringBuilds: true, // ✅ Bỏ qua lỗi ESLint khi build
  },
  typescript: {
    ignoreBuildErrors: true,  // ✅ Bỏ qua lỗi TypeScript khi build
  },
};

export default config;
