from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

# --- CORRECTED AGNO IMPORTS ---
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
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
GROQ_MODEL = "llama-3.3-70b-versatile" # Pass ID directly to model class

# Init Supabase
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Init Error: {e}")

# --- LAZY LOADING AI ---
team_instance = None
knowledge_base_instance = None

def get_ai_team():
    global team_instance, knowledge_base_instance
    if team_instance:
        return team_instance, knowledge_base_instance

    try:
        # 1. SETUP RAG
        if not SUPABASE_DB_URL or not OPENAI_API_KEY:
            print("Missing Database URL or OpenAI Key")
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
        # Note: Pass model object, not string to 'model' param usually
        groq_model = Groq(id=GROQ_MODEL)

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
        
        # Memory Agent
        memory_agent = Agent(
            name="Memory", role="Historian", model=groq_model,
            knowledge=knowledge_base_instance, 
            search_knowledge=True, 
            instructions=["Search the knowledge base for past context."]
        )

        # 3. DEFINE TEAM
        team_instance = Team(
            name="SuperTeam",
            agents=[tech_agent, data_agent, docs_agent, memory_agent],
            model=groq_model,
            instructions=[
                "You are the team leader.",
                "Delegate tasks to the appropriate agent based on the user query.",
                "1. History/Context -> Memory Agent",
                "2. Code/Debugging -> Tech Agent",
                "3. Data/Analysis -> Data Agent",
                "4. Writing/Docs -> Docs Agent",
                "If unsure, answer directly."
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
    # Note: 'def' runs in threadpool, preventing blocking of main loop
    try:
        team, kb = get_ai_team()
        
        if not team:
            raise HTTPException(status_code=500, detail="AI Team configuration failed.")

        # 1. Log User Message (Supabase)
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "user", "content": req.message
            }).execute()

        # 2. Run AI
        # Agno's team.run returns a RunResponse object
        response = team.run(req.message)
        
        # Extract content safely
        ai_content = response.content if hasattr(response, "content") else str(response)

        # 3. Log AI Response
        if supabase and req.conversation_id:
            supabase.table("chat_messages").insert({
                "conversation_id": req.conversation_id, "role": "ai", "content": ai_content
            }).execute()
            
            # OPTIONAL: Add to Knowledge Base (Heavy operation!)
            # Only do this if you want the AI to "learn" this conversation forever.
            # Ideally, offload this to a background task.
            if kb:
                kb.load_text(f"User: {req.message}\nAI: {ai_content}", upsert=True)

        return {"choices": [{"message": {"content": ai_content}}]}

    except Exception as e:
        # Print error to logs for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
