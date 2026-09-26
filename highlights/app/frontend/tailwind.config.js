/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [
    function ({ addVariant }) {
      addVariant("mob", ".mobile &");
      addVariant("dsk", ":not(.mobile) &");
    },
  ],
};
