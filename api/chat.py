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

# --- CUSTOM EMBEDDER FOR VERCEL ---
# This class replaces the heavy 'fastembed' library.
# It uses Hugging Face's API to generate embeddings on the fly.
try:
    from agno.knowledge.embedder.base import Embedder
except ImportError:
    # Fallback base class if agno library structure differs in production
    class Embedder:
        def __init__(self):
            self.dimensions = 384

class HuggingFaceServerlessEmbedder(Embedder):
    """
    Uses Hugging Face Inference API to generate embeddings without
    loading the model into memory. Perfect for Vercel Serverless.
    
    Default Model: BAAI/bge-small-en-v1.5
    (Matches FastEmbed default, ensuring vector compatibility)
    """
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", api_key: str = None):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        self.dimensions = 384 

    def get_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            print("WARNING: No HF_API_KEY found. RAG search will fail.")
            return [0.0] * self.dimensions
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(self.api_url, headers=headers, json={"inputs": text})
            if response.status_code != 200:
                print(f"HF Error: {response.text}")
                return [0.0] * self.dimensions
            
            # API returns a list of vectors. We take the first one.
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    return data[0] # List of lists
                elif isinstance(data[0], float):
                    return data # Single vector
            return [0.0] * self.dimensions
        except Exception as e:
            print(f"Embedding Error: {e}")
            return [0.0] * self.dimensions

# --- APP SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Flag to track if AI libraries loaded successfully
AI_AVAILABLE = False
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.groq import Groq
    from agno.vectordb.pgvector import PgVector, SearchType
    from agno.knowledge import Knowledge 
    AI_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI Library Import Error: {e}")

app = FastAPI(title="AI Agent Team API")

# CORS
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
POSTGRES_URL = os.environ.get("POSTGRES_URL") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Supabase DB Connection String construction
SUPABASE_DB_URL = None
if POSTGRES_URL:
    SUPABASE_DB_URL = POSTGRES_URL
else:
    # Fallback construction
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    db_name = os.environ.get("POSTGRES_DATABASE", "postgres")
    if all([db_user, db_password, db_host, db_name]):
        SUPABASE_DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

# Initialize Supabase Client (for Chat History logging)
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")

# Global Cache & ThreadPool
_team_cache = None
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
        return v.strip()

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

# --- CORE LOGIC ---
def initialize_team():
    """Initialize team with Serverless-compatible Knowledge Base."""
    global _team_cache
    
    if not AI_AVAILABLE:
        raise RuntimeError("AI libraries are not installed.")

    if _team_cache is not None:
        return _team_cache
    
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    
    groq_model = Groq(id=GROQ_MODEL_ID, api_key=GROQ_API_KEY)
    knowledge_base = None
    
    # Initialize knowledge base (Read-Only via Hugging Face Embeddings)
    if SUPABASE_DB_URL:
        try:
            # Use our custom Serverless Embedder (384 dim)
            embedder = HuggingFaceServerlessEmbedder()
            
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge",
                schema="public",
                embedder=embedder,
                search_type=SearchType.hybrid,
            )
            
            knowledge_base = Knowledge(vector_db=vector_db)
            
        except Exception as e:
            logger.warning(f"Knowledge Base initialization failed: {e}")
    
    # Instructions
    base_instructions = [
        "You are an expert in your field.",
        "Search the knowledge base using your tools to find relevant context.",
        "If the information is found in the knowledge base, answer the user's question accurately.",
        "CRITICAL: If the information is NOT found in the knowledge base or your internal knowledge, you MUST respond with: 'The information requested is out of scope.'",
        "Do not hallucinate or make up facts."
    ]

    # Agents
    tech_agent = Agent(
        name="Tech",
        role="Developer Support",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=base_instructions + ["Focus on code, architecture, and debugging."]
    )
    
    data_agent = Agent(
        name="Data",
        role="Data Analyst",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=base_instructions + ["Focus on database schemas, SQL, and data patterns."]
    )
    
    docs_agent = Agent(
        name="Docs",
        role="Technical Writer",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=base_instructions + ["Focus on documentation summaries and operational scope."]
    )
    
    # Supervisor
    team = Team(
        model=groq_model,
        members=[tech_agent, data_agent, docs_agent],
        instructions=[
            "You are a Supervisor. Analyze the user's query intent.",
            "Delegate the task to the single most appropriate agent (Tech, Data, or Docs).",
            "Do not answer the question yourself. Let the agent search the knowledge base.",
            "Return the agent's response directly."
        ]
    )
    
    _team_cache = team
    return team

def extract_agent_tag(content: str) -> tuple[str, str]:
    match = re.search(r"\[\[(TECH|DATA|DOCS|MEMORY|TEAM)\]\]", content)
    tag = "TEAM"
    cleaned_content = content
    if match:
        tag = match.group(1)
        cleaned_content = content.replace(match.group(0), "").strip()
    return tag, cleaned_content

# --- ENDPOINTS ---
@app.get("/health")
async def health_check():
    team_status = "active" if _team_cache else "uninitialized"
    return {"status": "healthy", "service": "AI Team API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(req: ChatRequest):
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI services unavailable.")

    try:
        loop = asyncio.get_event_loop()
        team = await loop.run_in_executor(executor, initialize_team)
        
        # 1. Log User Message (Async, optional)
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "user",
                    "content": req.message[:1000]
                }).execute()
            except Exception as e:
                logger.warning(f"Supabase logging error: {e}")

        # 2. Run AI (No History Injection, just current query)
        def run_agent_sync():
            return team.run(req.message)
        
        response = await loop.run_in_executor(executor, run_agent_sync)
        
        # 3. Process Response
        raw_content = response.content if hasattr(response, "content") else str(response)
        agent_tag, clean_content = extract_agent_tag(raw_content)
        
        # 4. Log AI Response (Async, optional)
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
