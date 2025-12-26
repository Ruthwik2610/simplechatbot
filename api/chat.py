from http.server import BaseHTTPRequestHandler
import json
import os
from agno.agent import Agent
from agno.team import Team
from supabase import create_client, Client

# --- CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") 

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Setup CORS
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        if self.command == 'OPTIONS':
            return

        try:
            # 2. Parse Request
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            user_message = body.get('message', '')
            conversation_id = body.get('conversation_id')

            # 3. Initialize Supabase
            previous_messages = []
            if conversation_id and SUPABASE_URL and SUPABASE_KEY:
                try:
                    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                    # Save user message
                    supabase.table("chat_messages").insert({
                        "conversation_id": conversation_id,
                        "role": "user",
                        "content": user_message
                    }).execute()
                    
                    # Fetch History (Limit 6 for efficiency)
                    history_response = supabase.table("chat_messages")\
                        .select("*")\
                        .eq("conversation_id", conversation_id)\
                        .order("created_at", desc=True)\
                        .limit(6)\
                        .execute()
                    
                    previous_messages = history_response.data[::-1]
                except Exception:
                    pass

            # 4. Context Engineering 
            formatted_history = ""
            if previous_messages:
                formatted_history += "<history>\n"
                for msg in previous_messages:
                    role_label = "User" if msg['role'] == 'user' else "Assistant"
                  
                    clean_content = msg['content'].replace('[[TECH]]', '').replace('[[DATA]]', '').replace('[[DOCS]]', '').replace('[[TEAM]]', '')
                    if "<file_content>" in clean_content:
                        clean_content = "[User attached file]"
                    formatted_history += f"{role_label}: {clean_content}\n"
                formatted_history += "</history>\n"

            # 5. Define Agents
            model_id = "groq:llama-3.1-8b-instant"

            tech_agent = Agent(
                name="Tech",
                role="Developer",
                model=model_id,
                instructions=["Fix code, debug, explain APIs."]
            )
            
            data_agent = Agent(
                name="Data", 
                role="Analyst",
                model=model_id, 
                instructions=["Analyze metrics, visualize trends."]
            )

            docs_agent = Agent(
                name="Docs", 
                role="Writer",
                model=model_id, 
                instructions=["Write summaries, SOPs, release notes."]
            )

            # 6. Parahelp-Style Optimized Instructions
            team = Team(
                model=model_id,
                members=[tech_agent, data_agent, docs_agent],
                show_tool_calls=False,
                instructions=[
                    "<role>Orchestrator</role>",
                    f"{formatted_history}",
                    "<logic>",
                    "IF query involves code/scripting/errors -> Delegate to Tech.",
                    "IF query involves analytics/charts/trends -> Delegate to Data.",
                    "IF query involves writing/summaries/SOPs -> Delegate to Docs.",
                    "IF query is greeting/general/unclear -> Answer as Team.",
                    "</logic>",
                    "<constraints>",
                    "1. Do NOT describe the plan. Execute delegation immediately.",
                    "2. Do NOT output internal xml tags in final response.",
                    "</constraints>",
                    "<output_format>",
                    "IMPORTANT: Prefix final response with EXACTLY one tag:",
                    "[[TECH]] <response> (if Tech used)",
                    "[[DATA]] <response> (if Data used)",
                    "[[DOCS]] <response> (if Docs used)",
                    "[[TEAM]] <response> (if self-answered)",
                    "</output_format>"
                ]
            )

            # 7. Execute
            response = team.run(user_message, stream=False)
            ai_content = response.content if hasattr(response, 'content') else str(response)

            # 8. Save AI Response
            if conversation_id and SUPABASE_URL and SUPABASE_KEY:
                try:
                    supabase.table("chat_messages").insert({
                        "conversation_id": conversation_id,
                        "role": "ai",
                        "content": ai_content
                    }).execute()
                except Exception:
                    pass

            # 9. Send Response
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": ai_content}}]
            }).encode('utf-8'))

        except Exception as e:
            print(f"Error: {str(e)}")
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
