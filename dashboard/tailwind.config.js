/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        terrain: {
          ground: "#8B9556",
          water: "#4A90D9",
          wall: "#6B6B6B",
          lava: "#D94A4A",
          ice: "#B0D4E8",
          sand: "#D4C07A",
          forest: "#2D6B3F",
          mountain: "#8B7355",
        },
        faction: {
          alpha: "#3B82F6",
          bravo: "#EF4444",
          charlie: "#10B981",
          delta: "#F59E0B",
        },
      },
    },
  },
  plugins: [],
};
