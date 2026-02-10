/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    50: '#eef7ff',
                    100: '#d9edff',
                    200: '#bce0ff',
                    300: '#8ecdff',
                    400: '#59b0ff',
                    500: '#338dff',
                    600: '#1a6cf5',
                    700: '#1357e1',
                    800: '#1646b6',
                    900: '#183e8f',
                    950: '#142757',
                },
                emerald: {
                    400: '#34d399',
                    500: '#10b981',
                },
                amber: {
                    400: '#fbbf24',
                    500: '#f59e0b',
                },
                rose: {
                    400: '#fb7185',
                    500: '#f43f5e',
                },
                surface: {
                    50: '#f8fafc',
                    100: '#f1f5f9',
                    200: '#e2e8f0',
                    700: '#1e293b',
                    800: '#0f172a',
                    900: '#0a0f1a',
                    950: '#060911',
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            boxShadow: {
                'glow-blue': '0 0 20px rgba(51, 141, 255, 0.3)',
                'glow-emerald': '0 0 20px rgba(16, 185, 129, 0.3)',
                'glow-amber': '0 0 20px rgba(245, 158, 11, 0.3)',
                'glow-rose': '0 0 20px rgba(244, 63, 94, 0.3)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'gradient-x': 'gradient-x 3s ease infinite',
                'fade-in': 'fadeIn 0.5s ease-out',
                'slide-up': 'slideUp 0.5s ease-out',
                'slide-in-left': 'slideInLeft 0.3s ease-out',
            },
            keyframes: {
                'gradient-x': {
                    '0%, 100%': { 'background-position': '0% 50%' },
                    '50%': { 'background-position': '100% 50%' },
                },
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { opacity: '0', transform: 'translateY(20px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                slideInLeft: {
                    '0%': { opacity: '0', transform: 'translateX(-20px)' },
                    '100%': { opacity: '1', transform: 'translateX(0)' },
                },
            },
        },
    },
    plugins: [],
}
