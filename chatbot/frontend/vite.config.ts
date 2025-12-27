import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// =============================================================================
// Vite Configuration — Premium CCA Chatbot Frontend
// =============================================================================
// Optimized for:
//   1. Fast HMR (Hot Module Replacement) during development
//   2. Production-optimized builds with code splitting
//   3. Proxy configuration for backend API communication
// =============================================================================

export default defineConfig({
    plugins: [react()],

    // ---------------------------------------------------------------------------
    // Path Resolution — Enable @/* imports
    // ---------------------------------------------------------------------------
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
        },
    },

    // ---------------------------------------------------------------------------
    // Development Server Configuration
    // ---------------------------------------------------------------------------
    server: {
        host: '0.0.0.0',
        port: 3000,
        strictPort: true,

        // Proxy API requests to backend during development
        proxy: {
            '/api': {
                target: 'http://host.docker.internal:8000',
                changeOrigin: true,
                secure: false,
            },
        },
    },

    // ---------------------------------------------------------------------------
    // Build Optimization
    // ---------------------------------------------------------------------------
    build: {
        target: 'esnext',
        minify: 'esbuild',
        sourcemap: false,

        // Code splitting for optimal caching
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    markdown: ['react-markdown', 'react-syntax-highlighter'],
                },
            },
        },
    },

    // ---------------------------------------------------------------------------
    // CSS Configuration
    // ---------------------------------------------------------------------------
    css: {
        devSourcemap: true,
    },
});
