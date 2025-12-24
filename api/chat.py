from http.server import BaseHTTPRequestHandler
import json
import os
from agno.agent import Agent
from agno.team import Team
from supabase import create_client, Client

# --- CONFIGURATION ---
# Ensure these are set in your Vercel Environment Variables
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

            if not conversation_id:
                self.wfile.write(json.dumps({'error': 'Missing conversation_id'}).encode('utf-8'))
                return

            # 3. Initialize Supabase Client
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

            # 4. Save User Message to DB
            supabase.table("chat_messages").insert({
                "conversation_id": conversation_id,
                "role": "user",
                "content": user_message
            }).execute()

            # 5. Fetch Chat History (Context Engineering)
            # Get last 10 messages for this conversation
            history_response = supabase.table("chat_messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=True)\
                .limit(10)\
                .execute()
            
            # Reverse to chronological order (Oldest -> Newest)
            previous_messages = history_response.data[::-1]
            
            formatted_history = "\n<conversation_history>\n"
            for msg in previous_messages:
                role_label = "User" if msg['role'] == 'user' else "Assistant"
                # Scrub file content placeholders if needed to save tokens
                clean_content = msg['content']
                if "<file_content>" in clean_content:
                    clean_content = "[User attached a file]"
                formatted_history += f"{role_label}: {clean_content}\n"
            formatted_history += "</conversation_history>\n"

            # 6. Define Agents
            model_id = "groq:llama-3.1-8b-instant"

            tech_agent = Agent(
                name="tech",
                model=model_id,
                instructions=["You are a Technical Expert.", "Answer with code examples."]
            )
            
            data_agent = Agent(
                name="data", 
                instructions="You are a Data Analyst.", 
                model=model_id
            )

            docs_agent = Agent(
                name="docs", 
                instructions="You are a Documentation Writer.", 
                model=model_id
            )

            # 7. Run Team with History
            team = Team(
                model=model_id,
                members=[tech_agent, data_agent, docs_agent],
                instructions=[
                    "You are the Intelligent Routing Orchestrator.",
                    f"CONTEXT: Use the following history to understand the conversation:\n{formatted_history}",
                    "ROUTING TAGS: Start response with [[TECH]], [[DATA]], [[DOCS]], or [[TEAM]]."
                ]
            )

            response = team.run(user_message, stream=False)
            ai_content = response.content if hasattr(response, 'content') else str(response)

            # 8. Save AI Response to DB
            supabase.table("chat_messages").insert({
                "conversation_id": conversation_id,
                "role": "ai",
                "content": ai_content
            }).execute()

            # 9. Send Response to Client
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": ai_content}}]
            }).encode('utf-8'))

        except Exception as e:
            print(f"Error: {str(e)}")
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
