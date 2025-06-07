// project/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Tells Tailwind to scan all your React components for CSS classes
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}