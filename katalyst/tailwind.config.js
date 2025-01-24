/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#4CAF50",
        border: "#000000",
        background: "#FFFFFF",
        surface: {
          DEFAULT: "#FFFFFF",
          hover: "#F5F5F5",
        },
      },
    },
  },
  plugins: [],
};
