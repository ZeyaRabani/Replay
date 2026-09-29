/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // single warm accent — everything else stays tailwind default
        amber: {
          200: "#ffe1a8", 300: "#ffcf70", 400: "#f5b432",
          500: "#e0a020", 600: "#b8810f", 700: "#8a5f08",
          900: "#3d2a05",
        },
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
