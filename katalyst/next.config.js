/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  typescript: {
    // Enable TypeScript features
    ignoreBuildErrors: false,
  },
}

module.exports = nextConfig
