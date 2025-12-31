from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator  # <-- ADD THIS IMPORT
from typing import Optional, List
import os
import json
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORTS ---
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.groq import Groq
    from agno.vectordb.pgvector import PgVector, SearchType
    from agno.knowledge.knowledge import Knowledge 
    from agno.knowledge.embedder.openai import OpenAIEmbedder
    from supabase import create_client, Client
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create dummy classes for testing
    class Agent: pass
    class Team: pass
    class Groq: pass
    class PgVector: pass
    class Knowledge: pass
    class OpenAIEmbedder: pass
    class Client: pass

app = FastAPI()

# --- CORS ---
# For production, restrict origins
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Vercel provides these environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
POSTGRES_URL = os.environ.get("POSTGRES_URL")  # Vercel provides this directly
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Make sure this is set!

# Use POSTGRES_URL if available, otherwise construct it
if POSTGRES_URL:
    SUPABASE_DB_URL = POSTGRES_URL.replace("postgresql://", "postgresql://")  # Ensure proper scheme
else:
    # Fallback to individual variables
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    db_name = os.environ.get("POSTGRES_DATABASE", "postgres")
    
    if all([db_user, db_password, db_host, db_name]):
        SUPABASE_DB_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    else:
        SUPABASE_DB_URL = None
        logger.warning("No PostgreSQL URL configured")

# Groq model
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Initialize clients
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")

# Cache the team to avoid reinitialization on every request
_team_cache = None
_knowledge_base_cache = None

def initialize_team():
    """Initialize team once and cache it"""
    global _team_cache, _knowledge_base_cache
    
    if _team_cache is not None:
        return _team_cache, _knowledge_base_cache
    
    logger.info("Initializing AI team...")
    
    # Check for required API keys
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set!")
        raise RuntimeError("GROQ_API_KEY is required")
    
    # Initialize model
    groq_model = Groq(id=GROQ_MODEL_ID, api_key=GROQ_API_KEY)
    
    knowledge_base = None
    
    # Initialize knowledge base if we have DB URL and OpenAI key
    if SUPABASE_DB_URL and OPENAI_API_KEY:
        try:
            logger.info("Initializing knowledge base...")
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge",
                schema="public",
                embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
                search_type=SearchType.hybrid,
            )
            knowledge_base = Knowledge(vector_db=vector_db)
            logger.info("Knowledge base initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}")
    else:
        logger.info("Running without knowledge base (simple mode)")
    
    # Define agents
    tech_agent = Agent(
        name="Tech",
        role="Developer",
        model=groq_model,
        instructions=["Provide code in markdown.", "Debug errors clearly."]
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
    
    # Memory agent (only if knowledge base exists)
    memory_agent = None
    if knowledge_base:
        memory_agent = Agent(
            name="Memory",
            role="Historian",
            model=groq_model,
            knowledge=knowledge_base,
            search_knowledge=True,
            instructions=["Search the knowledge base for past context."]
        )
    
    # Create team with available agents
    members = [tech_agent, data_agent, docs_agent]
    if memory_agent:
        members.append(memory_agent)
    
    team = Team(
        model=groq_model,
        members=members,
        instructions=[
            "Analyze intent and delegate.",
            "1. If context/history needed -> Memory Agent (if available)",
            "2. If Code -> Tech Agent",
            "3. If Analysis -> Data Agent",
            "4. If Writing -> Docs Agent",
            "PREFIX response with [[TECH]], [[DATA]], [[DOCS]], [[MEMORY]], or [[TEAM]]."
        ]
    )
    
    _team_cache = team
    _knowledge_base_cache = knowledge_base
    
    return team, knowledge_base

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) > 5000:
            raise ValueError('Message too long')
        return v.strip()

class ChatResponse(BaseModel):
    content: str
    agent: str = "TEAM"

@app.get("/")
async def root():
    return {"status": "healthy", "service": "AI Chat API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        team, kb = initialize_team()
        supabase_status = "connected" if supabase else "disconnected"
        kb_status = "enabled" if kb else "disabled"
        
        return {
            "status": "healthy",
            "supabase": supabase_status,
            "knowledge_base": kb_status,
            "team_initialized": team is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "supabase": "error",
            "knowledge_base": "error"
        }

@app.post("/api/chat")
async def chat_handler(req: ChatRequest):
    """Handle chat requests"""
    try:
        # Initialize team (cached after first call)
        team, knowledge_base = initialize_team()
        
        # Log user message
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "user",
                    "content": req.message[:1000]  # Limit length for storage
                }).execute()
            except Exception as e:
                logger.warning(f"Failed to log user message: {e}")
        
        # Get recent history
        history_context = ""
        if supabase and req.conversation_id:
            try:
                hist = supabase.table("chat_messages") \
                    .select("*") \
                    .eq("conversation_id", req.conversation_id) \
                    .order("created_at", desc=True) \
                    .limit(6) \
                    .execute()
                
                if hist.data:
                    # Reverse to get chronological order
                    msgs = hist.data[::-1]
                    history_context = "\n<recent_chat_history>\n" + \
                                    "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + \
                                    "\n</recent_chat_history>\n"
            except Exception as e:
                logger.warning(f"Failed to fetch history: {e}")
        
        # Run AI
        full_prompt = f"{history_context}\nUser Query: {req.message}"
        response = team.run(full_prompt)
        
        # Extract content and agent tag
        ai_content = response.content if hasattr(response, "content") else str(response)
        
        # Parse agent tag from response
        agent_tag = "TEAM"
        for tag in ["[[TECH]]", "[[DATA]]", "[[DOCS]]", "[[MEMORY]]", "[[TEAM]]"]:
            if tag in ai_content:
                agent_tag = tag.strip("[]")
                break
        
        # Clean up the tag from content if it's at the beginning
        for tag in ["[[TECH]]", "[[DATA]]", "[[DOCS]]", "[[MEMORY]]", "[[TEAM]]"]:
            if ai_content.startswith(tag):
                ai_content = ai_content[len(tag):].strip()
                break
        
        # Log AI response
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "ai",
                    "content": ai_content[:2000]  # Limit length
                }).execute()
            except Exception as e:
                logger.warning(f"Failed to log AI response: {e}")
        
        return {
            "choices": [{
                "message": {
                    "content": ai_content,
                    "role": "ai"
                }
            }],
            "agent": agent_tag
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e)[:200]  # Limit error message length
            }
        )

# Add startup event
@app.on_event("startup")
async def startup():
    """Initialize resources on startup"""
    logger.info("Starting up AI Chat API...")
    try:
        # Warm up the team (calls initialize_team which caches)
        team, kb = initialize_team()
        logger.info(f"Team initialized. Knowledge base: {'Enabled' if kb else 'Disabled'}")
    except Exception as e:
        logger.error(f"Failed to initialize team on startup: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
