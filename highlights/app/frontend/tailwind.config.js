/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Premier League palette — existing class names re-map onto it
        zinc: {
          950: "#1a001e", 900: "#25002a", 800: "#37003C",
          700: "#4b1a52", 600: "#613467", 500: "#8f6f95",
          400: "#b49db9", 300: "#d3c5d6", 200: "#e7dee9",
          100: "#f6f1f7", 50: "#fbf8fb",
        },
        amber: {
          200: "#ffb3cf", 300: "#ff5f97", 400: "#E90052",
          500: "#c80047", 600: "#a8003c", 700: "#82002f",
          900: "#4d001c",
        },
        emerald: {
          100: "#ccffe7", 200: "#8affca", 300: "#00FF85",
          400: "#00e676", 700: "#007a44", 800: "#005a35",
          900: "#003f27",
        },
        sky: { 200: "#b3fcff", 300: "#04F5FF", 700: "#00777d" },
        violet: { 200: "#e0c3ff", 300: "#c08cff", 800: "#5a2a99" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [
    function ({ addVariant }) {
      addVariant("mob", ".mobile &");
      addVariant("dsk", "html:not(.mobile) &");
    },
  ],
};
