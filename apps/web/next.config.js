/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://localhost:8000/api/:path*" },
      { source: "/data/demo/:path*", destination: "http://localhost:8000/data/demo/:path*" },
    ];
  },
};
module.exports = nextConfig;
