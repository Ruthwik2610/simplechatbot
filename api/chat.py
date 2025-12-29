from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

# --- IMPORTS ---
from agno.agent import Agent
from agno.team import Team
from agno.knowledge import Knowledge  # Correct class name
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.openai import OpenAIEmbedder
from supabase import create_client, Client

app = FastAPI()

# --- CORS (Crucial) ---
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
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
GROQ_MODEL = "groq:llama-3.3-70b-versatile"

# Init Supabase for Chat Logs
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except:
        pass

# --- LAZY LOADING AI (Prevents Startup Crashes) ---
team_instance = None
knowledge_base_instance = None

def get_ai_team():
    global team_instance, knowledge_base_instance
    if team_instance:
        return team_instance, knowledge_base_instance

    try:
        # 1. SETUP RAG (Knowledge Base)
        if not SUPABASE_DB_URL or not OPENAI_API_KEY:
            return None, None

        vector_db = PgVector(
            db_url=SUPABASE_DB_URL,
            table_name="agent_knowledge",
            schema="public",
            embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
            search_type=SearchType.hybrid, 
        )
        knowledge_base_instance = Knowledge(vector_db=vector_db)

        # 2. DEFINE AGENTS
        tech_agent = Agent(
            name="Tech", role="Developer", model=GROQ_MODEL,
            instructions=["Provide code in markdown.", "Debug errors clearly."]
        )
        data_agent = Agent(
            name="Data", role="Analyst", model=GROQ_MODEL,
            instructions=["Analyze patterns.", "Suggest visualizations."]
        )
        docs_agent = Agent(
            name="Docs", role="Writer", model=GROQ_MODEL,
            instructions=["Write clear summaries.", "Use professional formatting."]
        )
        
        # Memory Agent (The one that searches DB)
        memory_agent = Agent(
            name="Memory", role="Historian", model=GROQ_MODEL,
            knowledge=knowledge_base_instance, 
            search_knowledge=True, 
            instructions=["Search the knowledge base for past context and summarize it."]
        )

        # 3. DEFINE SUPERVISOR
        team_instance = Team(
            model=GROQ_MODEL,
            members=[tech_agent, data_agent, docs_agent, memory_agent],
            instructions=[
                "<role>Orchestrator</role>",
                "<logic>",
                "Analyze intent.",
                "1. If history/recall needed -> Delegate to 'Memory'.",
                "2. If Code -> Delegate to 'Tech'.",
                "3. If Analysis -> Delegate to 'Data'.",
                "4. If Writing -> Delegate to 'Docs'.",
                "5. Else -> Answer as 'Team'.",
                "</logic>",
                "<formatting>Prefix response with [[TECH]], [[DATA]], [[DOCS]], [[MEMORY]], or [[TEAM]].</formatting>"
            ]
        )
        return team_instance, knowledge_base_instance

    except Exception as e:
        print(f"AI Config Error: {e}")
        return None, None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@app.post("/api/chat")
def chat_handler(req: ChatRequest):
    try:
        team, kb = get_ai_team()
        
        if not team:
            return {"choices": [{"message": {"content": "[[TEAM]] System Error: AI keys missing in Vercel."}}]}

        # 1. Save User Message & Embed It
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "user", "content": req.message
            }).execute()
            if kb: kb.load_text(req.message, metadata={"conversation_id": req.conversation_id, "role": "user"})

        # 2. Run AI
        response = team.run(f"User Query: {req.message}")
        ai_content = response.content if hasattr(response, "content") else str(response)

        # 3. Save AI Response & Embed It
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "ai", "content": ai_content
            }).execute()
            if kb: kb.load_text(ai_content, metadata={"conversation_id": req.conversation_id, "role": "ai"})

        return {"choices": [{"message": {"content": ai_content}}]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
