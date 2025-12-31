import os
import logging
import asyncio
import re
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- IMPORTS & FALLBACKS ---
# We use a flag to track if AI features are available
AI_AVAILABLE = False
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.groq import Groq
    from agno.vectordb.pgvector import PgVector, SearchType
    from agno.knowledge.knowledge import Knowledge 
    from agno.knowledge.embedder.openai import OpenAIEmbedder
    AI_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI Library Import Error: {e}. AI features will be disabled.")

app = FastAPI(title="AI Agent Team API")

# --- CORS ---
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
POSTGRES_URL = os.environ.get("POSTGRES_URL") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Construct DB URL
SUPABASE_DB_URL = None
if POSTGRES_URL:
    SUPABASE_DB_URL = POSTGRES_URL.replace("postgresql://", "postgresql://")
else:
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    db_name = os.environ.get("POSTGRES_DATABASE", "postgres")
    
    if all([db_user, db_password, db_host, db_name]):
        SUPABASE_DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    else:
        logger.warning("No PostgreSQL connection details found.")

# Initialize Supabase Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")

# Global Cache
_team_cache = None
_knowledge_base_cache = None

# ThreadPool for blocking AI calls
executor = ThreadPoolExecutor(max_workers=5)

# --- MODELS ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) > 5000:
            raise ValueError('Message too long')
        return v.strip()

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

# --- CORE LOGIC ---
def initialize_team():
    """Initialize team once and cache it. Blocking function."""
    global _team_cache, _knowledge_base_cache
    
    if not AI_AVAILABLE:
        raise RuntimeError("AI libraries (agno) are not installed.")

    if _team_cache is not None:
        return _team_cache, _knowledge_base_cache
    
    logger.info("Initializing AI team configuration...")
    
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    
    groq_model = Groq(id=GROQ_MODEL_ID, api_key=GROQ_API_KEY)
    knowledge_base = None
    
    # Initialize knowledge base
    if SUPABASE_DB_URL and OPENAI_API_KEY:
        try:
            logger.info("Connecting to Knowledge Base...")
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge",
                schema="public",
                embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
                search_type=SearchType.hybrid,
            )
            knowledge_base = Knowledge(vector_db=vector_db)
            # Optional: knowledge_base.load() if needed
        except Exception as e:
            logger.warning(f"Knowledge Base initialization failed: {e}")
    
    # Define Agents
    tech_agent = Agent(
        name="Tech",
        role="Developer",
        model=groq_model,
        instructions=["Provide code in markdown.", "Debug errors clearly.", "Be concise."]
    )
    
    data_agent = Agent(
        name="Data",
        role="Analyst",
        model=groq_model,
        instructions=["Analyze patterns.", "Suggest visualizations."]
    )
    
    docs_agent = Agent(
        name="Docs",
        role="Writer",
        model=groq_model,
        instructions=["Write clear summaries.", "Use professional formatting."]
    )
    
    # Memory Agent
    memory_agent = None
    if knowledge_base:
        memory_agent = Agent(
            name="Memory",
            role="Historian",
            model=groq_model,
            knowledge=knowledge_base,
            search_knowledge=True,
            instructions=["Search the knowledge base for past context provided."]
        )
    
    members = [tech_agent, data_agent, docs_agent]
    if memory_agent:
        members.append(memory_agent)
    
    team = Team(
        model=groq_model,
        members=members,
        instructions=[
    "You are a router. Route the user query to the single most appropriate agent immediately.",
    "Do not check Memory unless the user specifically asks about 'previous' or 'last' conversation.",
    "If the user wants a summary, send directly to Docs Agent.",
    "Do not add your own commentary, just return the agent's response."
]
    )
    
    _team_cache = team
    _knowledge_base_cache = knowledge_base
    return team, knowledge_base

def extract_agent_tag(content: str) -> tuple[str, str]:
    """Extracts agent tag using Regex and cleans content."""
    # Matches [[TAG]]
    match = re.search(r"\[\[(TECH|DATA|DOCS|MEMORY|TEAM)\]\]", content)
    tag = "TEAM"
    cleaned_content = content
    
    if match:
        tag = match.group(1)
        # Remove the tag from the content
        cleaned_content = content.replace(match.group(0), "").strip()
        
    return tag, cleaned_content

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "healthy", "service": "AI Agent Team API"}

@app.get("/health")
async def health_check():
    team_status = "active" if _team_cache else "uninitialized"
    if not AI_AVAILABLE:
        team_status = "disabled_missing_lib"
        
    return {
        "status": "healthy",
        "supabase": "connected" if supabase else "disconnected",
        "team": team_status
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(req: ChatRequest):
    """Async chat handler that offloads blocking AI calls to thread."""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI services are unavailable on this server.")

    try:
        # 1. Initialize (or get cached) Team
        # We run this in executor to avoid blocking if init takes time (DB connections)
        loop = asyncio.get_event_loop()
        team, _ = await loop.run_in_executor(executor, initialize_team)
        
        # 2. Retrieve History (Async I/O)
        history_context = ""
        if supabase and req.conversation_id:
            try:
                # Log User Message
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "user",
                    "content": req.message[:1000]
                }).execute()
                
                # Fetch Context
                hist = supabase.table("chat_messages") \
                    .select("*") \
                    .eq("conversation_id", req.conversation_id) \
                    .order("created_at", desc=True) \
                    .limit(6) \
                    .execute()
                
                if hist.data:
                    msgs = hist.data[::-1] # Reverse for chronological order
                    history_context = "\n<recent_chat_history>\n" + \
                        "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + \
                        "\n</recent_chat_history>\n"
            except Exception as e:
                logger.warning(f"Supabase history error: {e}")

        # 3. Execute AI (Blocking Call -> Offloaded)
        full_prompt = f"{history_context}\nUser Query: {req.message}"
        
        def run_agent_sync():
            return team.run(full_prompt)
        
        # Run blocking AI task in thread pool
        response = await loop.run_in_executor(executor, run_agent_sync)
        
        # 4. Process Response
        raw_content = response.content if hasattr(response, "content") else str(response)
        agent_tag, clean_content = extract_agent_tag(raw_content)
        
        # 5. Log AI Response (Async I/O)
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "ai",
                    "content": clean_content[:2000]
                }).execute()
            except Exception as e:
                logger.warning(f"Failed to log AI response: {e}")

        return ChatResponse(content=clean_content, agent=agent_tag)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    """Warmup on startup"""
    if AI_AVAILABLE and GROQ_API_KEY:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, initialize_team)
            logger.info("Startup: Team initialized successfully")
        except Exception as e:
            logger.warning(f"Startup initialization warning: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
