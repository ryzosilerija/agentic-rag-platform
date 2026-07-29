/** @type {import(''next'').NextConfig} */
const nextConfig = {
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://localhost:8010";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;