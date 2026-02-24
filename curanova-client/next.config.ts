import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow images from any source (captured camera images use data: URLs)
  images: {
    unoptimized: true,
  },
  // Proxy API/WebSocket requests to the Python FastAPI server during development
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/ws/:path*",
        destination: `${apiUrl}/ws/:path*`,
      },
    ];
  },
};

export default nextConfig;
