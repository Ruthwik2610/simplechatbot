from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json

# --- AGNO & SUPABASE IMPORTS ---
from agno.agent import Agent
from agno.team import Team
from supabase import create_client, Client

# --- INIT FASTAPI ---
app = FastAPI()

# --- CORS SETTINGS ---
# Crucial for Vercel: Allow your frontend to hit this backend
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
MODEL1 = "groq:llama-3.1-8b-instant"
MODEL2="groq:llama-3.1-70b-versatile"

# Initialize Supabase Client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- DEFINE AGENTS (Global Scope for Efficiency) ---
# Tech Agent: Handles code execution, debugging, and technical concepts
tech_agent = Agent(
    name="Tech",
    role="Developer",
    model=MODEL1,
    instructions=[
        "You are a senior software engineer.",
        "Provide code snippets in markdown blocks.",
        "Debug errors and explain solutions clearly."
    ]
)

# Data Agent: Handles analytics, math, and trends
data_agent = Agent(
    name="Data",
    role="Analyst",
    model=MODEL1,
    instructions=[
        "You are a data analyst.",
        "Analyze text or data patterns.",
        "Suggest visualizations or metrics."
    ]
)

# Docs Agent: Handles writing and documentation
docs_agent = Agent(
    name="Docs",
    role="Writer",
    model=MODEL1,
    instructions=[
        "You are a technical writer.",
        "Write clear summaries, SOPs, and documentation.",
        "Use professional, easy-to-read formatting."
    ]
)

# Team Orchestrator: Decides which agent to use
team = Team(
    model=MODEL2,
    members=[tech_agent, data_agent, docs_agent],
    instructions=[
        "<role>Orchestrator</role>",
        "<logic>",
        "Analyze the user's input to determine the intent.",
        "IF intent is Code/Debugging/Scripting -> Delegate to 'Tech'.",
        "IF intent is Analysis/Math/Charts -> Delegate to 'Data'.",
        "IF intent is Writing/Documentation/Summary -> Delegate to 'Docs'.",
        "IF intent is Greeting/General -> Answer directly as 'Team'.",
        "</logic>",
        "<formatting>",
        "You MUST prefix your final response with ONE of these tags:",
        "[[TECH]] - if Tech agent was used.",
        "[[DATA]] - if Data agent was used.",
        "[[DOCS]] - if Docs agent was used.",
        "[[TEAM]] - if you answered directly.",
        "Example: [[TECH]] Here is the python code fix...",
        "</formatting>"
    ]
)

# --- REQUEST MODEL ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# --- MAIN ENDPOINT ---
# Note: The route matches the path in your chat.js fetch call
@app.post("/api/chat")
def chat_handler(req: ChatRequest):
    try:
        # 1. Save User Message to Supabase
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "user",
                    "content": req.message
                }).execute()
            except Exception as e:
                print(f"Supabase Insert Error: {e}")

        # 2. Fetch Context (History)
        history_context = ""
        if supabase and req.conversation_id:
            try:
                # Fetch last 6 messages
                hist_res = supabase.table("chat_messages")\
                    .select("*")\
                    .eq("conversation_id", req.conversation_id)\
                    .order("created_at", desc=True)\
                    .limit(6)\
                    .execute()
                
                # Reverse to get chronological order
                msgs = hist_res.data[::-1]
                
                if msgs:
                    history_context = "<history>\n"
                    for m in msgs:
                        # Clean tags from history so model doesn't get confused
                        clean_content = m['content'].replace('[[TECH]]', '').replace('[[DATA]]', '').replace('[[DOCS]]', '')
                        role = "User" if m['role'] == 'user' else "Assistant"
                        history_context += f"{role}: {clean_content}\n"
                    history_context += "</history>\n"
            except Exception as e:
                print(f"Supabase History Error: {e}")

        # 3. Run the AI Team
        # We append history to the prompt so the agent sees it
        full_prompt = f"{history_context}\nUser Query: {req.message}"
        
        # Run agent (agno team run)
        response = team.run(full_prompt)
        
        # specific handling depending on what 'team.run' returns
        # Assuming it returns an object with a .content string or is a string itself
        ai_content = response.content if hasattr(response, "content") else str(response)

        # 4. Save AI Response to Supabase
        if supabase and req.conversation_id:
            try:
                supabase.table("chat_messages").insert({
                    "conversation_id": req.conversation_id,
                    "role": "ai",
                    "content": ai_content
                }).execute()
            except Exception as e:
                print(f"Supabase Save AI Error: {e}")

        # 5. Return JSON in the format chat.js expects
        return {
            "choices": [
                {
                    "message": {
                        "content": ai_content
                    }
                }
            ]
        }

    except Exception as e:
        print(f"SERVER ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/api/health")
def health():
    return {"status": "ok", "backend": "FastAPI + Agno"}
