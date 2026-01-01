import os
import logging
import asyncio
import re
import requests
import urllib.parse
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from supabase import create_client, Client

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NexusFlow")

# --- 2. CUSTOM EMBEDDER (VERCEL COMPATIBLE) ---
try:
    from agno.knowledge.embedder.base import Embedder
except ImportError:
    class Embedder:
        def __init__(self): self.dimensions = 384

class HuggingFaceServerlessEmbedder(Embedder):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        super().__init__()
        self.model = model
        self.api_key = os.environ.get("HF_API_KEY", "").strip()
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        self.dimensions = 384 

    def get_embedding(self, text: str) -> List[float]:
        if not self.api_key: return [0.0] * self.dimensions
        try:
            res = requests.post(self.api_url, headers={"Authorization": f"Bearer {self.api_key}"}, json={"inputs": text})
            if res.status_code != 200: return [0.0] * self.dimensions
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0] if isinstance(data[0], list) else data
            return [0.0] * self.dimensions
        except:
            return [0.0] * self.dimensions

# --- 3. AGNO AI SETUP ---
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.groq import Groq
    from agno.vectordb.pgvector import PgVector, SearchType
    from agno.knowledge import Knowledge 
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.error("Agno AI libraries not found.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. ROBUST CONNECTION STRING FIXER ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Fetch URL
POSTGRES_URL = os.environ.get("POSTGRES_URL", "")

SUPABASE_DB_URL = None

if POSTGRES_URL:
    # NUCLEAR FIX: Remove ALL whitespace from the string. 
    # This fixes the "invalid connection option 'supa'" error caused by spaces.
    clean_url = "".join(POSTGRES_URL.split())
    # Fix protocol
    SUPABASE_DB_URL = clean_url.replace("postgres://", "postgresql://", 1)
else:
    # Manual Construction Fallback
    db_user = os.environ.get("POSTGRES_USER", "postgres").strip()
    db_pass = urllib.parse.quote_plus(os.environ.get("POSTGRES_PASSWORD", "").strip())
    db_host = os.environ.get("POSTGRES_HOST", "").strip()
    db_name = os.environ.get("POSTGRES_DATABASE", "postgres").strip()
    
    if db_host:
        port = "5432"
        if ":" in db_host: db_host, port = db_host.split(":")
        SUPABASE_DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{port}/{db_name}"

# Ensure SSL mode is set if using Supabase Pooler
if SUPABASE_DB_URL and "sslmode" not in SUPABASE_DB_URL:
    separator = "&" if "?" in SUPABASE_DB_URL else "?"
    SUPABASE_DB_URL += f"{separator}sslmode=require"

# Initialize Supabase HTTP Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: pass

_team_cache = None
executor = ThreadPoolExecutor(max_workers=5)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

# --- 5. TEAM INITIALIZATION ---
def initialize_team():
    global _team_cache
    if _team_cache: return _team_cache
    if not AI_AVAILABLE: raise RuntimeError("AI libraries missing")
    
    groq_model = Groq(id=GROQ_MODEL, api_key=GROQ_API_KEY)
    knowledge_base = None

    # Connect to Vector DB
    if SUPABASE_DB_URL:
        try:
            logger.info(f"Connecting to DB...") 
            embedder = HuggingFaceServerlessEmbedder()
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge",
                schema="public",
                embedder=embedder,
                search_type=SearchType.hybrid,
            )
            knowledge_base = Knowledge(vector_db=vector_db)
            logger.info("✅ Knowledge Base Connected")
        except Exception as e:
            # Log the error but DO NOT crash the app. Allow chat to work without KB.
            logger.error(f"KB Connection Failed: {e}")
            knowledge_base = None

    # Dynamic Instructions
    if knowledge_base:
        instructions = [
            "Search the knowledge base for context.",
            "If information is NOT found in the KB, strictly reply: 'The information requested is out of scope.'",
            "Do not hallucinate."
        ]
    else:
        instructions = [
            "NOTICE: The knowledge base is currently unavailable (Database Error).",
            "Answer generic questions if possible, but inform the user you cannot check documentation."
        ]

    # Agents
    # search_knowledge must be False if knowledge_base is None to prevent "tool_use_failed" errors
    tech_agent = Agent(
        name="Tech",
        role="Developer Support",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=bool(knowledge_base),
        instructions=instructions
    )
    
    data_agent = Agent(
        name="Data",
        role="Data Analyst",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=bool(knowledge_base),
        instructions=instructions
    )
    
    docs_agent = Agent(
        name="Docs",
        role="Technical Writer",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=bool(knowledge_base),
        instructions=instructions
    )
    
    team = Team(
        model=groq_model,
        members=[tech_agent, data_agent, docs_agent],
        instructions=["Delegate to the most appropriate agent based on the user query."]
    )
    
    _team_cache = team
    return team

def extract_agent_tag(content: str):
    match = re.search(r"\[\[(TECH|DATA|DOCS|MEMORY|TEAM)\]\]", content)
    tag = "TEAM"
    cleaned = content
    if match:
        tag = match.group(1)
        cleaned = content.replace(match.group(0), "").strip()
    return tag, cleaned

# --- 6. ENDPOINT ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(req: ChatRequest):
    if not AI_AVAILABLE: raise HTTPException(503, "AI Services Unavailable")
    
    try:
        loop = asyncio.get_event_loop()
        team = await loop.run_in_executor(executor, initialize_team)
        
        # Log User
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "user",
                    "content": req.message[:1000]
                }).execute()
            except: pass

        # Run AI
        def run_sync(): return team.run(req.message)
        response = await loop.run_in_executor(executor, run_sync)
        
        # Process Output
        raw = response.content if hasattr(response, "content") else str(response)
        tag, clean = extract_agent_tag(raw)
        
        # Log AI
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "ai",
                    "content": clean[:2000]
                }).execute()
            except: pass

        return ChatResponse(content=clean, agent=tag)

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return ChatResponse(content=f"System Error: {str(e)}", agent="ERROR")
