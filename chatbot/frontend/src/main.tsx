// =============================================================================
// main.tsx — Application Entry Point
// =============================================================================
// React 18 with StrictMode for development checks
// =============================================================================

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Get root element
const rootElement = document.getElementById('root');

if (!rootElement) {
    throw new Error('Root element not found. Ensure index.html has a div with id="root".');
}

// Create React 18 root and render
createRoot(rootElement).render(
    <StrictMode>
        <App />
    </StrictMode>
);
