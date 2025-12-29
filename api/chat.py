from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json

# --- AGNO & SUPABASE IMPORTS ---
from agno.agent import Agent
from agno.team import Team
from agno.knowledge import Knowledge  
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.openai import OpenAIEmbedder
from supabase import create_client, Client

# --- INIT FASTAPI ---
app = FastAPI()

# --- CORS (Crucial for Vercel) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Ensure these are set in Vercel Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL") # Format: postgresql+psycopg://user:pass@host:port/postgres
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
GROQ_MODEL = "groq:llama-3.3-70b-versatile"

# Initialize Supabase Client (For simple logging)
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 1. SETUP VECTOR MEMORY (RAG) ---
# This connects the agents to the 'agent_knowledge' table
vector_db = PgVector(
    db_url=SUPABASE_DB_URL,
    table_name="agent_knowledge",
    schema="public",
    embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
    search_type=SearchType.hybrid, 
)

knowledge_base = =Knowledge(vector_db=vector_db)


# --- 2. DEFINE AGENTS ---

# Tech Agent
tech_agent = Agent(
    name="Tech",
    role="Developer",
    model=GROQ_MODEL,
    instructions=["Provide code in markdown.", "Debug errors clearly."]
)

# Data Agent
data_agent = Agent(
    name="Data",
    role="Analyst",
    model=GROQ_MODEL,
    instructions=["Analyze patterns.", "Suggest visualizations."]
)

# Docs Agent
docs_agent = Agent(
    name="Docs",
    role="Writer",
    model=GROQ_MODEL,
    instructions=["Write clear summaries and SOPs.", "Use professional formatting."]
)

# --- NEW: MEMORY AGENT (The Summarizer) ---
# This agent has exclusive access to the Vector DB
memory_agent = Agent(
    name="Memory",
    role="Historian",
    model=GROQ_MODEL,
    knowledge=knowledge_base, 
    search_knowledge=True, # Enables RAG lookup
    instructions=[
        "You are the memory of this conversation.",
        "Search the knowledge base for past context.",
        "Summarize what was discussed previously based on the user's query.",
        "If the user asks 'what did we talk about', summarize the recent vector history."
    ]
)

# --- SUPERVISOR TEAM ---
team = Team(
    model=GROQ_MODEL,
    members=[tech_agent, data_agent, docs_agent, memory_agent],
    instructions=[
        "<role>Orchestrator</role>",
        "<logic>",
        "Analyze the user's input to determine the intent.",
        "1. IF the user asks to SUMMARIZE the conversation, recall history, or asks 'what did I say' -> Delegate to 'Memory'.",
        "2. IF intent is Code/Debugging -> Delegate to 'Tech'.",
        "3. IF intent is Analysis/Math -> Delegate to 'Data'.",
        "4. IF intent is Writing/Docs -> Delegate to 'Docs'.",
        "5. IF intent is Greeting -> Answer directly as 'Team'.",
        "</logic>",
        "<formatting>",
        "Prefix response with ONE tag:",
        "[[TECH]]", "[[DATA]]", "[[DOCS]]", "[[MEMORY]]", "[[TEAM]]",
        "</formatting>"
    ]
)

# --- MODELS ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

# --- ENDPOINTS ---

@app.post("/api/login")
def login_handler(creds: LoginRequest):
    # Securely read users.json from the API folder
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "users.json")
        
        with open(json_path, "r") as f:
            users = json.load(f)
            
        user = next((u for u in users if u["email"] == creds.email and u["password"] == creds.password), None)
        
        if user:
            return {"success": True, "user": {"email": user["email"], "name": user["name"]}}
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password")
            
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="User database not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_handler(req: ChatRequest):
    try:
        # 1. Save User Message to Supabase Logs (UI History)
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id,
                "role": "user",
                "content": req.message
            }).execute()

            # --- AUTO-LEARN: Save to Vector DB for Future Recall ---
            try:
                knowledge_base.load_text(
                    text=req.message, 
                    metadata={"conversation_id": req.conversation_id, "role": "user"}
                )
            except Exception as v_err:
                print(f"Vector Save Error: {v_err}")

        # 2. Run the AI Team (Supervisor decides who answers)
        # Note: We send just the query. If 'Memory' is selected, it fetches history itself.
        response = team.run(f"User Query: {req.message}")
        
        ai_content = response.content if hasattr(response, "content") else str(response)

        # 3. Save AI Response to Supabase Logs
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id,
                "role": "ai",
                "content": ai_content
            }).execute()

            # --- AUTO-LEARN: Save AI Answer to Vector DB ---
            try:
                knowledge_base.load_text(
                    text=ai_content, 
                    metadata={"conversation_id": req.conversation_id, "role": "ai"}
                )
            except Exception as v_err:
                print(f"Vector Save AI Error: {v_err}")

        return {
            "choices": [{"message": {"content": ai_content}}]
        }

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok", "backend": "FastAPI + Agno + Supabase Vectors"}
