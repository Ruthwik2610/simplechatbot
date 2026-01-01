import os
import logging
import asyncio
import re
import requests
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from supabase import create_client, Client

# --- CUSTOM EMBEDDER (Vercel Compatible) ---
try:
    from agno.knowledge.embedder.base import Embedder
except ImportError:
    class Embedder:
        def __init__(self):
            self.dimensions = 384

class HuggingFaceServerlessEmbedder(Embedder):
    """
    Uses Hugging Face Inference API. 
    Default Model: BAAI/bge-small-en-v1.5 (Matches FastEmbed vectors)
    """
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", api_key: str = None):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        self.dimensions = 384 

    def get_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            print("WARNING: No HF_API_KEY found.")
            return [0.0] * self.dimensions
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(self.api_url, headers=headers, json={"inputs": text})
            if response.status_code != 200:
                print(f"HF Error: {response.text}")
                return [0.0] * self.dimensions
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list): return data[0]
                elif isinstance(data[0], float): return data
            return [0.0] * self.dimensions
        except Exception as e:
            print(f"Embedding Error: {e}")
            return [0.0] * self.dimensions

# --- APP SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AI_AVAILABLE = False
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.groq import Groq
    from agno.vectordb.pgvector import PgVector, SearchType
    from agno.knowledge import Knowledge 
    AI_AVAILABLE = True
except ImportError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
POSTGRES_URL = os.environ.get("POSTGRES_URL") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- FIX 1: AUTO-CORRECT DB URL ---
SUPABASE_DB_URL = None
if POSTGRES_URL:
    SUPABASE_DB_URL = POSTGRES_URL
else:
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    db_name = os.environ.get("POSTGRES_DATABASE", "postgres")
    if all([db_user, db_password, db_host, db_name]):
        SUPABASE_DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

# CRITICAL FIX: SQLAlchemy requires 'postgresql://', NOT 'postgres://'
if SUPABASE_DB_URL and SUPABASE_DB_URL.startswith("postgres://"):
    SUPABASE_DB_URL = SUPABASE_DB_URL.replace("postgres://", "postgresql://", 1)

# Initialize Supabase Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Supabase Client Error: {e}")

_team_cache = None
executor = ThreadPoolExecutor(max_workers=5)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

def initialize_team():
    global _team_cache
    if not AI_AVAILABLE: raise RuntimeError("AI libraries missing")
    if _team_cache: return _team_cache
    
    groq_model = Groq(id=GROQ_MODEL_ID, api_key=GROQ_API_KEY)
    
    knowledge_base = None
    has_kb = False

    # Initialize Knowledge Base
    if SUPABASE_DB_URL:
        try:
            logger.info("Attempting to connect to Knowledge Base...")
            embedder = HuggingFaceServerlessEmbedder()
            
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge",
                schema="public",
                embedder=embedder,
                search_type=SearchType.hybrid,
            )
            
            knowledge_base = Knowledge(vector_db=vector_db)
            has_kb = True
            logger.info("✅ Knowledge Base Connected Successfully")
            
        except Exception as e:
            logger.error(f"❌ Knowledge Base Initialization FAILED: {e}")
            # We continue without KB, but flags below will handle it

    # --- FIX 2: DYNAMIC INSTRUCTIONS ---
    # Only tell the agent to search if the KB actually loaded
    if has_kb:
        instructions = [
            "Search the knowledge base for context.",
            "If information is not found in the knowledge base, strictly reply: 'The information requested is out of scope.'",
            "Do not hallucinate."
        ]
    else:
        instructions = [
            "NOTICE: The knowledge base is currently unavailable.",
            "Inform the user you cannot access internal documentation right now.",
            "Do NOT attempt to use any search tools."
        ]

    # Initialize Agents with conditional search_knowledge flag
    tech_agent = Agent(
        name="Tech",
        role="Developer Support",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=has_kb, # CRITICAL: Only True if KB exists
        instructions=instructions
    )
    
    data_agent = Agent(
        name="Data",
        role="Data Analyst",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=has_kb,
        instructions=instructions
    )
    
    docs_agent = Agent(
        name="Docs",
        role="Technical Writer",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=has_kb,
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(req: ChatRequest):
    if not AI_AVAILABLE: raise HTTPException(status_code=503, detail="AI unavailable")
    
    try:
        loop = asyncio.get_event_loop()
        team = await loop.run_in_executor(executor, initialize_team)
        
        # 1. Log User
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({"conversation_id": req.conversation_id, "role": "user", "content": req.message[:1000]}).execute()
            except: pass

        # 2. Run Model
        def run_sync(): return team.run(req.message)
        response = await loop.run_in_executor(executor, run_sync)
        
        # 3. Process
        raw = response.content if hasattr(response, "content") else str(response)
        tag, clean = extract_agent_tag(raw)
        
        # 4. Log AI
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({"conversation_id": req.conversation_id, "role": "ai", "content": clean[:2000]}).execute()
            except: pass

        return ChatResponse(content=clean, agent=tag)
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        # Return a clean error to the frontend instead of 500 crash
        return ChatResponse(content=f"System Error: {str(e)}", agent="ERROR")
