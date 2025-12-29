from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

# --- IMPORTS ---
from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.knowledge import Knowledge 
from agno.knowledge.embedder.openai import OpenAIEmbedder
from supabase import create_client, Client

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# 1. Standard API Keys (What you ALREADY have)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

# 2. RAG Keys (What you need for Vectors)
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# Init Supabase Client (For Chat Logs)
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Client Error: {e}")

# --- AI SETUP ---
def get_team():
    """
    Builds the AI Team. 
    If DB_URL is present -> Enables RAG (Knowledge Base).
    If missing -> Falls back to Simple Mode (No RAG).
    """
    
    groq_model = Groq(id=GROQ_MODEL_ID)
    knowledge_base = None

    # --- TRY CONNECTING TO KNOWLEDGE BASE ---
    if SUPABASE_DB_URL and OPENAI_API_KEY:
        try:
            vector_db = PgVector(
                db_url=SUPABASE_DB_URL,
                table_name="agent_knowledge", # Make sure this table exists in Supabase
                schema="public",
                embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
                search_type=SearchType.hybrid, 
            )
            knowledge_base = Knowledge(vector_db=vector_db)
            print("SUCCESS: RAG Knowledge Base Connected.")
        except Exception as e:
            print(f"WARNING: RAG Failed to load (Check DB URL). Running in Simple Mode. Error: {e}")
    else:
        print("NOTICE: Missing SUPABASE_DB_URL or OPENAI_API_KEY. Running in Simple Mode.")

    # --- DEFINE AGENTS ---
    tech_agent = Agent(
        name="Tech", role="Developer", model=groq_model,
        instructions=["Provide code in markdown.", "Debug errors clearly."]
    )
    
    data_agent = Agent(
        name="Data", role="Analyst", model=groq_model,
        instructions=["Analyze patterns.", "Suggest visualizations."]
    )

    docs_agent = Agent(
        name="Docs", role="Writer", model=groq_model,
        instructions=["Write clear summaries.", "Use professional formatting."]
    )

    # Memory Agent (Only useful if Knowledge Base works)
    memory_agent = Agent(
        name="Memory", role="Historian", model=groq_model,
        knowledge=knowledge_base,
        search_knowledge=True if knowledge_base else False, 
        instructions=["Search the knowledge base for past context."]
    )

    # --- DEFINE TEAM ---
    team = Team(
        model=groq_model,
        members=[tech_agent, data_agent, docs_agent, memory_agent],
        instructions=[
            "<role>Orchestrator</role>",
            "Analyze intent and delegate.",
            "1. If context/history needed -> Memory Agent",
            "2. If Code -> Tech Agent",
            "3. If Analysis -> Data Agent",
            "4. If Writing -> Docs Agent",
            "PREFIX response with [[TECH]], [[DATA]], [[DOCS]], [[MEMORY]], or [[TEAM]]."
        ]
    )
    return team, knowledge_base

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@app.post("/api/chat")
def chat_handler(req: ChatRequest):
    try:
        # 1. Save User Msg to Supabase (Simple Log)
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "user", "content": req.message
            }).execute()

        # 2. Get Context (Simple History Fetch - Works without RAG)
        history_context = ""
        if supabase and req.conversation_id:
            try:
                hist = supabase.table("chat_messages").select("*").eq("conversation_id", req.conversation_id).order("created_at", desc=True).limit(6).execute()
                msgs = hist.data[::-1] # Reverse order
                if msgs:
                    history_context = "\n<recent_chat_history>\n" + "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + "\n</recent_chat_history>\n"
            except: 
                pass

        # 3. Run AI
        team, kb = get_team()
        
        # If we have a Knowledge Base, we can search it (RAG)
        # Note: We DON'T load the chat into KB here to avoid pollution
        
        full_prompt = f"{history_context}\nUser Query: {req.message}"
        response = team.run(full_prompt)
        ai_content = response.content if hasattr(response, "content") else str(response)

        # 4. Save AI Msg to Supabase (Simple Log)
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "ai", "content": ai_content
            }).execute()

        return {"choices": [{"message": {"content": ai_content}}]}

    except Exception as e:
        print(f"Error: {e}")
        # Return a fallback message if it crashes
        return {"choices": [{"message": {"content": f"[[TEAM]] System Error: {str(e)}"}}]}
