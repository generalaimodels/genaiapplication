# main.py
# --------------------------------------------------------------------------------------------------
# SYSTEM: Advanced Generalized RAG Backend API
# AUTHOR: IQ-300 Implementation
# STACK: FastAPI, Asyncio, Pydantic, SQLAlchemy(SQLite), Custom AI Modules
# PATTERNS: CQRS-lite, Dependency Injection, Worker Pools, Asynchronous Event Loop, RAG Pipeline
# --------------------------------------------------------------------------------------------------

import os
import time
import uuid
import json
import asyncio
import shutil
import logging
import sqlite3
import dataclasses
from typing import List, Optional, Dict, Any, Generator, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

# Third-party imports
import uvicorn
import torch
from fastapi import (
    FastAPI, HTTPException, Depends, Request, 
    UploadFile, File, BackgroundTasks, status, Form
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------------------------------------------------------
# INTERNAL MODULE IMPORTS (Based on provided context)
# --------------------------------------------------------------------------------------------------
from conversion import UltraDoclingPipeline, PipelineConfig
from history import GeneralizedChatHistory, Role
from torchchuck import chunk_from_files, ChunkConfig
from rethinker_retrieval import EmbeddingAdapter, VectorBase, Rethinker, IVFBuildParams, IVFSearchParams, RethinkerParams
# Note: Renamed vllm_generation.py import to match usage. Assuming file is vllm_generation.py
from vllm_generation import LLMClient, LLMConfig, PromptTemplate, ChatResult

# --------------------------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("API")

UPLOAD_DIR = "./temp_uploads"
ARTIFACTS_DIR = "./data_artifacts"
DB_PATH = "./api_logs.db"

# Hardware detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_CORES = os.cpu_count() or 4
# Pool for heavy CPU tasks (PDF OCR, Chunking)
PROCESS_POOL = ProcessPoolExecutor(max_workers=max(1, CPU_CORES - 2))
# Pool for I/O bound tasks or bridging sync libraries
THREAD_POOL = ThreadPoolExecutor(max_workers=CPU_CORES * 5)

# --------------------------------------------------------------------------------------------------
# PERSISTENCE LAYER: API LOGGING (Request/Response/Session)
# --------------------------------------------------------------------------------------------------
def init_api_db():
    """Initializes the SQLite database for request auditing and session tracking."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Session Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        uuid TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        meta_data TEXT
    )""")
    
    # History/Interaction Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_uuid TEXT,
        session_uuid TEXT,
        endpoint TEXT,
        query TEXT,
        response TEXT,
        latency_ms REAL,
        status_code INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_uuid) REFERENCES sessions(uuid)
    )""")
    
    conn.commit()
    conn.close()

def log_interaction(req_uuid: str, session_uuid: str, endpoint: str, query: str, response: str, latency: float, status: int):
    """Asynchronous logging helper (fire-and-forget in background task)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO interactions (request_uuid, session_uuid, endpoint, query, response, latency_ms, status_code) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (req_uuid, session_uuid, endpoint, query, response, latency, status)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

# --------------------------------------------------------------------------------------------------
# GLOBAL SINGLETONS (LIFESPAN STATE)
# --------------------------------------------------------------------------------------------------
class GlobalState:
    llm_client: Optional[LLMClient] = None
    vector_base: Optional[VectorBase] = None
    history_engine: Optional[GeneralizedChatHistory] = None
    rethinker: Optional[Rethinker] = None

state = GlobalState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Handles startup initialization of heavy AI models and connection pools.
    """
    logger.info("--- SYSTEM STARTUP ---")
    
    # 1. Initialize API DB
    init_api_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 2. Initialize LLM Client
    # NOTE: In production, secrets come from ENV.
    llm_cfg = LLMConfig(
        api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("LLM_MODEL", "meta-llama/Llama-3-70b-Instruct"),
        max_concurrency=32
    )
    state.llm_client = LLMClient(llm_cfg)
    logger.info(f"LLM Client initialized: {llm_cfg.model}")

    # 3. Initialize Embedding & VectorBase
    # We load the embedding model once to VRAM.
    hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    adapter = EmbeddingAdapter(hf_emb, device=DEVICE, normalize=True)
    state.vector_base = VectorBase(adapter, dim=768, metric="cosine", device=DEVICE)
    
    # Create or Load Collection (Simplified logic for demo: creates purely in-memory or loads specific file)
    # Ideally, VectorBase should support persistence. Here we ensure the structure exists.
    try:
        state.vector_base.create_collection("global_docs", 768, "cosine")
        logger.info("VectorBase collection 'global_docs' ready.")
    except Exception as e:
        logger.warning(f"VectorBase initialization note: {e}")

    # 4. Initialize Rethinker (Reranker)
    # Parameters tuned for high-precision RAG
    rethinker_params = RethinkerParams(
        seed_sem_topk=30,
        max_depth=2, 
        w_sem_query=0.7, 
        w_lex=0.3,
        top_nodes_final=5
    )
    state.rethinker = Rethinker(state.vector_base, rethinker_params)

    # 5. Initialize Chat History Engine
    state.history_engine = GeneralizedChatHistory(db_folder=ARTIFACTS_DIR, d=768)
    logger.info("History Engine ready.")

    yield
    
    logger.info("--- SYSTEM SHUTDOWN ---")
    if state.history_engine:
        state.history_engine.close()
    PROCESS_POOL.shutdown(wait=False)
    THREAD_POOL.shutdown(wait=False)

# --------------------------------------------------------------------------------------------------
# PYDANTIC MODELS (API CONTRACTS)
# --------------------------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    runtime: str
    device: str

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique session ID")
    message: str = Field(..., min_length=1)
    branch_id: str = Field(default="main")
    model_override: Optional[str] = None
    temperature: Optional[float] = 0.7

class Reference(BaseModel):
    doc_id: str
    content_snippet: str
    score: float

class ChatResponse(BaseModel):
    request_uuid: str
    conversation_id: str
    role: str = "assistant"
    content: str
    references: List[Reference]
    latency_s: float

class IngestResponse(BaseModel):
    task_id: str
    status: str
    file_name: str
    message: str

# --------------------------------------------------------------------------------------------------
# APP INITIALIZATION & MIDDLEWARE
# --------------------------------------------------------------------------------------------------

app = FastAPI(
    title="Ultra-Advanced RAG API",
    version="1.0.0",
    description="High-performance, concurrent AI backend.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS: Security Best Practice - Restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: Request Timing & ID Injection
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    request.state.uuid = req_id
    
    try:
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        response.headers["X-Request-ID"] = req_id
        
        # NOTE: Logging happens in the endpoint or background tasks to capture body content
        # which is consumed in middleware otherwise.
        return response
    except Exception as e:
        logger.error(f"Critical Middleware Error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"detail": "Internal Server Error", "uuid": req_id}
        )

# --------------------------------------------------------------------------------------------------
# CORE LOGIC: HEAVY COMPUTATION WRAPPERS
# --------------------------------------------------------------------------------------------------

def _cpu_bound_ingest(file_path: str, output_dir: str):
    """
    Executes the Conversion -> Chunking -> vector prep pipeline.
    Must run in ProcessPool to avoid GIL blocking.
    """
    try:
        # 1. Conversion
        cfg = PipelineConfig(
            images_scale=2.0, do_ocr=True, ocr_engine="tesseract_cli",
            ocr_languages=["en"], accelerator_device="auto"
        )
        pipeline = UltraDoclingPipeline(input_path=file_path, output_dir=output_dir, config=cfg)
        # Assuming these methods exist on the instance as per prompt
        # pipeline.extract_images()
        # pipeline.export_tables()
        pipeline.ocr_to_markdown() # Generates markdown files in output_dir
        
        # 2. Chunking
        chunk_cfg = ChunkConfig(
            chunk_size=1024, chunk_overlap=128, separators=("\n\n", "\n", " ", ""),
            keep_separator="end", device=None, distributed=False, return_offsets=True,
            dedup=True, compile=False, normalize_ws=True, max_space_run=1,
            max_newline_run=2, strip_line_edges=True, normalize_punct_runs=True,
            max_punct_run=3, strip_ascii_bars=True, bar_min_len=6,
            punct_chars="._-~=*#", num_workers_io=2, num_workers_chunk=2,
            drop_blank_chunks=True
        )
        # chunk_from_files expects an input path (directory or file)
        chunks = chunk_from_files(output_dir, chunk_cfg)
        
        # 3. Format for VectorBase (flatten generator)
        # VectorBase.insert expects dicts with 'text' and 'metadata' usually.
        # Mapping chunk tuple/object to dict
        records = []
        for c in chunks:
            # Assuming chunk object has text and metadata attributes or is a dict
            txt = c.text if hasattr(c, 'text') else str(c)
            meta = c.metadata if hasattr(c, 'metadata') else {"source": file_path}
            records.append({"text": txt, **meta})
            
        return records
    except Exception as e:
        logger.error(f"Ingest failed for {file_path}: {e}")
        raise e

async def process_document_background(file_path: str, req_uuid: str):
    """
    Orchestrator for background ingestion.
    1. Runs CPU bound tasks in ProcessPool.
    2. Updates Vector Index (Thread safe/GPU bound).
    """
    logger.info(f"[{req_uuid}] Starting background ingestion: {file_path}")
    loop = asyncio.get_running_loop()
    
    try:
        # Offload CPU heavy lifting
        output_dir = os.path.join(ARTIFACTS_DIR, f"proc_{req_uuid}")
        records = await loop.run_in_executor(PROCESS_POOL, _cpu_bound_ingest, file_path, output_dir)
        
        if not records:
            logger.warning(f"[{req_uuid}] No chunks generated.")
            return

        # Update Vector DB (GPU operation, safe to run in thread or main loop if async supported)
        # VectorBase operations are likely synchronous PyTorch/FAISS wrappers, so offload to ThreadPool
        def _update_index():
            state.vector_base.insert(records)
            # Rebuild index if necessary (IVF requires training, usually done once or periodically)
            # For dynamic updates, we assume a Flat index or HNSW buffer
            if not state.vector_base.is_indexed:
                 state.vector_base.build_index("IVF_OPQ_PQ", IVFBuildParams(nlist=64, pq_m=16))
            return len(records)

        count = await loop.run_in_executor(THREAD_POOL, _update_index)
        
        # Log success
        log_interaction(req_uuid, "system", "ingest_worker", f"Processed {file_path}", f"Inserted {count} chunks", 0.0, 200)
        logger.info(f"[{req_uuid}] Ingestion complete. {count} chunks added.")

    except Exception as e:
        logger.error(f"[{req_uuid}] Ingestion Error: {e}")
        log_interaction(req_uuid, "system", "ingest_worker", f"Processed {file_path}", f"Error: {str(e)}", 0.0, 500)
    finally:
        # Cleanup temp files
        if os.path.exists(file_path):
            os.remove(file_path)
        # shutil.rmtree(output_dir, ignore_errors=True) # Optional: keep for debug

# --------------------------------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System diagnostic endpoint."""
    return HealthResponse(
        status="active",
        runtime=os.getenv("ENV", "production"),
        device=DEVICE
    )

@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...)
):
    """
    Endpoint: Ingest PDF/Document.
    Logic: Save file -> return 202 -> Process in background (OCR -> Chunk -> Embed -> Index).
    """
    req_uuid = request.state.uuid
    
    # Validate file type
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Only PDF and Images supported currently.")

    # Save to temp disk (Async I/O)
    temp_path = os.path.join(UPLOAD_DIR, f"{req_uuid}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")
    finally:
        file.file.close()

    # Trigger background task
    background_tasks.add_task(process_document_background, temp_path, req_uuid)

    return IngestResponse(
        task_id=req_uuid,
        status="processing_started",
        file_name=file.filename,
        message="Document queued for conversion and indexing."
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Endpoint: RAG Chat.
    Logic:
    1. Retrieve Conversation History.
    2. Vector Search (Semantic) + Reranking (Rethinker).
    3. Construct Prompt (History + Context + Query).
    4. Generate LLM Response.
    5. Save new interaction to History.
    """
    start_ts = time.perf_counter()
    req_uuid = request.state.uuid
    
    # 1. Update History with User Message
    # We use timestamps for ordering.
    user_ts = time.time()
    state.history_engine.add_message(
        payload.conversation_id, payload.branch_id, 
        msg_id=hash(f"{user_ts}{payload.message}"), # Simple hash for demo
        role=Role.USER, 
        content=payload.message, 
        ts=user_ts,
        tokens=len(payload.message)//4 # Rough est
    )

    # 2. Retrieve Context (Hybrid Search + Rerank)
    # Using Rethinker which combines VectorBase search + Logic
    # We run this in thread pool as it might contain blocking Pytorch/numpy calls
    loop = asyncio.get_running_loop()
    
    def _retrieve_context():
        # Using Rethinker's search
        search_result = state.rethinker.search(payload.message)
        return search_result.get("contexts", [])

    contexts = await loop.run_in_executor(THREAD_POOL, _retrieve_context)
    
    # Format Context for Prompt
    context_str = "\n\n".join([f"Source {i+1}: {c['text']}" for i, c in enumerate(contexts)])
    
    # 3. Retrieve Historical Context (Last N messages)
    # Using history engine's build_context
    history_ctx = state.history_engine.build_context(
        payload.conversation_id, payload.branch_id, 
        query_text=payload.message, 
        budget_ctx=2048
    )
    # Convert history tuples to string format
    history_str = "\n".join([f"{row[3]}" for row in history_ctx]) # row[3] is content based on example

    # 4. Construct Prompt
    system_prompt = (
        "You are an advanced AI assistant. Use the provided Context and Chat History to answer the user's query. "
        "If the answer is not in the context, admit it. Be technical and precise."
    )
    
    user_prompt_template = PromptTemplate(
        "Context:\n{context}\n\nHistory:\n{history}\n\nUser Query: {query}"
    )
    
    final_user_prompt = user_prompt_template.render({
        "context": context_str,
        "history": history_str,
        "query": payload.message
    })

    # 5. Generate Response (Async LLM)
    try:
        result: ChatResult = await state.llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=final_user_prompt,
            temperature=payload.temperature,
            model=payload.model_override
        )
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        raise HTTPException(status_code=502, detail="LLM Provider Error")

    # 6. Save Assistant Response to History
    asst_ts = time.time()
    state.history_engine.add_message(
        payload.conversation_id, payload.branch_id,
        msg_id=hash(f"{asst_ts}{result.content}"),
        role=Role.ASSISTANT,
        content=result.content,
        ts=asst_ts,
        tokens=len(result.content)//4,
        parent_msg_id=hash(f"{user_ts}{payload.message}") # simplified linkage
    )

    # 7. Log Interaction (Async Side-effect)
    total_time = time.perf_counter() - start_ts
    background_tasks.add_task(
        log_interaction, 
        req_uuid, payload.conversation_id, "/chat", payload.message, result.content, total_time, 200
    )

    # 8. Construct Response
    refs = []
    for i, ctx in enumerate(contexts):
        # Handle metadata extraction safely
        meta = ctx.get('metadata', {})
        refs.append(Reference(
            doc_id=meta.get('source', 'unknown'),
            content_snippet=ctx.get('text', '')[:100] + "...",
            score=ctx.get('score', 0.0)
        ))

    return ChatResponse(
        request_uuid=req_uuid,
        conversation_id=payload.conversation_id,
        content=result.content,
        references=refs,
        latency_s=total_time
    )

@app.delete("/api/v1/history/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(conversation_id: str):
    """
    Hard delete a conversation branch.
    NOTE: The history module snippet provided doesn't show a delete method, 
    so we assume direct DB manipulation or extending the class. 
    Here we just implement the API stub.
    """
    # Placeholder: In real impl, call state.history_engine.delete_conversation(conversation_id)
    logger.info(f"Request to clear history for {conversation_id}")
    return None

# --------------------------------------------------------------------------------------------------
# EXCEPTION HANDLING
# --------------------------------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Error [{request.state.uuid}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "type": type(exc).__name__,
            "request_uuid": getattr(request.state, "uuid", "unknown")
        }
    )

if __name__ == "__main__":
    # Production entry point: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)