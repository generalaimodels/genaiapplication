#  Chatbot

The  Chatbot is a production-grade, full-stack AI conversational agent featuring a FastAPI backend, React frontend, and robust RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Quick Start Guide

Follow these steps to set up the project from scratch.

### Prerequisites
- **Python** 3.10 or higher
- **Node.js** 18 or higher
- **Docker** & **Docker Compose** (Optional, for containerized run)
- **Git**

---

### Method 1: Local Development Setup (Recommended)

#### 1. Backend Setup

1. **Navigate to the project root:**
   ```bash
   cd /path/to/chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration (.env):**
   Create a `.env` file in the root directory if you need to override defaults (e.g., for external LLM providers).
   ```bash
   # Example .env content
   LLM_BASE_URL=http://localhost:8007/v1
   LLM_API_KEY=your-api-key
   LLM_MODEL=openai/gpt-oss-20b
   ```

5. **Start the Backend Server:**
   ```bash
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *The backend documentation will be available at `http://localhost:8000/docs`.*

#### 2. Frontend Setup

1. **Open a new terminal** and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. **Install Node modules:**
   ```bash
   npm install
   ```

3. **Start the Development Server:**
   ```bash
   npm run dev
   ```

4. **Access the App:**
   Open your browser and visit `http://localhost:3000`.

---

### Method 2: Docker Setup (Easiest)

If you have Docker installed, you can run the entire stack with a single command.

1. **Build and Run:**
   ```bash
   docker compose up --build
   ```

   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

---

## 🛠 Features

- **Real-time Streaming:** Token-by-token streaming response with visual indicators.
- **RAG (Retrieval-Augmented Generation):** Chat with your documents.
- **Math & LaTeX Support:** Full rendering of mathematical equations (`$$...$$`, `\(...\)`).
- **Session Management:** History tracking, auto-naming, and persistence.
- **Feedback Loop:** User feedback mechanism for model improvement.

## 📂 Project Structure

- **`api/`**: FastAPI backend source code.
  - **`routers/`**: API endpoints (chat, sessions, documents).
  - **`database.py`**: SQLite database management.
- **`frontend/`**: React application.
  - **`src/`**: Components, hooks, and API clients.
- **`vllm_generation.py`**: High-performance LLM client wrapper.

## 🤝 Troubleshooting

- **Streaming Issues:** Ensure your LLM provider supports streaming and the `LLM_BASE_URL` is correct.
- **Math Not Rendering:** Refresh the page to ensure latest styling is loaded.
- **Dependencies:** If `pip install` fails on `torch`, try installing it manually with the correct CUDA/ROCm version for your hardware.

```bash
docker builder prune -a -f && docker compose build --no-cache && docker compose up
```
```bash
DOCKER_BUILDKIT=0 docker compose up --build
```