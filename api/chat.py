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
from pydantic import BaseModel
from supabase import create_client, Client

# --- 1. LOGGING & SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NexusFlow")

# --- 2. VALIDATION HELPER ---
def validate_postgres_url(url: Optional[str]) -> str:
    if not url: return None
    url = url.strip()
    if " " in url: return None
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url

# --- 3. CUSTOM EMBEDDER (VERCEL COMPATIBLE) ---
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

# --- 4. AGNO AI SETUP ---
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

# --- 5. ENV VARS & DB SETUP ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Initialize Supabase REST Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: pass

# Initialize Database URL
raw_pg_url = os.environ.get("POSTGRES_URL")
SUPABASE_DB_URL = validate_postgres_url(raw_pg_url)

_team_cache = None
executor = ThreadPoolExecutor(max_workers=5)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

# --- 6. ACTION TOOLS (The "Hands") ---
# These simulate the write-actions described in PDF pages 3-5.

def run_diagnostics_action(system: str) -> str:
    """Simulates running a diagnostic check on a system (VPN, Gateway, etc)."""
    return f"DIAGNOSTIC COMPLETE: {system} is responding. Latency: 45ms. Status: Online."

def unlock_account_action(username: str) -> str:
    """Action to unlock a user account in the system."""
    return f"SUCCESS: Account for '{username}' has been UNLOCKED. Notification sent to user."

def process_refund_action(invoice_id: str) -> str:
    """Action to process a refund for a specific invoice."""
    return f"SUCCESS: Refund processed for Invoice {invoice_id}. Amount will appear in 3-5 days."

# --- 7. TEAM LOGIC ---
def initialize_team():
    global _team_cache
    if _team_cache: return _team_cache
    if not AI_AVAILABLE: raise RuntimeError("AI libraries missing")
    
    groq_model = Groq(id=GROQ_MODEL, api_key=GROQ_API_KEY)
    knowledge_base = None

    # Connect to Knowledge Base (Your Uploaded CSV Data)
    if SUPABASE_DB_URL:
        try:
            logger.info("Initializing PgVector...")
            embedder = HuggingFaceServerlessEmbedder()
            
            db_url = SUPABASE_DB_URL
            if db_url and "&supa=" in db_url:
                db_url = db_url.split("&supa=")[0]

            if db_url:
                vector_db = PgVector(
                    db_url=db_url,
                    table_name="agent_knowledge", # This matches your ingestion script
                    schema="public",
                    embedder=embedder,
                    search_type=SearchType.hybrid,
                )
                knowledge_base = Knowledge(vector_db=vector_db)
                logger.info("✅ Knowledge Base Connected")
            else:
                knowledge_base = None

        except Exception as e:
            logger.error(f"KB Connection Failed: {e}")
            knowledge_base = None

    # --- DEFINE SPECIALIZED AGENTS (Matching PDF Architecture) ---

    # 1. Technical Support Agent ("The Troubleshooter")
    # Reference: PDF Page 2 & 3
    tech_agent = Agent(
        name="Technical Support",
        role="IT Support Specialist",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True, # MUST search for "System Log" entries
        tools=[run_diagnostics_action],
        instructions=[
            "You are the Tier 1 IT Support technician.",
            "If a user reports an error, FIRST search the knowledge base for 'System Log' or 'Error Code' to find recent server issues.",
            "Use the 'run_diagnostics_action' tool only if no known outage is found in the logs.",
            "Append [[SUPPORT]] to your final response."
        ]
    )
    
    # 2. Account & Access Agent ("The Gatekeeper")
    # Reference: PDF Page 3 & 4
    access_agent = Agent(
        name="Access Control",
        role="IAM Administrator",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True, # MUST search for "User Record"
        tools=[unlock_account_action],
        instructions=[
            "You handle account locks and passwords.",
            "You are STRICT about identity verification.",
            "If a user asks to unlock an account:",
            "1. Search the knowledge base for 'User Directory Record' or the user's name.",
            "2. Verify if the account status is actually 'Locked'.",
            "3. Ask the user for their Employee ID.",
            "4. compare the provided ID with the 'ID' found in the knowledge base.",
            "5. ONLY if they match, call 'unlock_account_action'.",
            "Append [[ACCESS]] to your final response."
        ]
    )
    
    # 3. Billing & Subscription Agent ("The Finance Clerk")
    # Reference: PDF Page 4 & 5
    billing_agent = Agent(
        name="Billing",
        role="Finance Specialist",
        model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True, # MUST search for "Financial Record"
        tools=[process_refund_action],
        instructions=[
            "You handle refunds and invoices.",
            "Refunds are strictly limited to 30 days from the 'Last Payment Date'.",
            "If a user requests a refund:",
            "1. Search the knowledge base for 'Financial Record' or 'Invoice' linked to the user.",
            "2. Check the 'Last Payment Date' and calculate if it is within 30 days.",
            "3. If >30 days, politely deny the request citing policy.",
            "4. If <30 days, call 'process_refund_action'.",
            "Append [[BILLING]] to your final response."
        ]
    )
    
    # Supervisor (Orchestrator)
    team = Team(
        model=groq_model,
        members=[tech_agent, access_agent, billing_agent],
        instructions=[
            "You are the IT Service Supervisor.",
            "Route the user's ticket to the correct specialist based on their request.",
            "Hardware/VPN/Errors -> Technical Support",
            "Login/Passwords/Access -> Access Control",
            "Money/Invoices/Refunds -> Billing"
        ]
    )
    
    _team_cache = team
    return team

def extract_agent_tag(content: str):
    # Updated regex to capture the new PDF-aligned tags
    match = re.search(r"\[\[(SUPPORT|ACCESS|BILLING|TEAM)\]\]", content)
    tag = "TEAM"
    cleaned = content
    if match:
        tag = match.group(1)
        cleaned = content.replace(match.group(0), "").strip()
    return tag, cleaned

# --- 8. ENDPOINT ---
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
