from http.server import BaseHTTPRequestHandler
import json
import os
from agno.agent import Agent
from agno.team import Team

# NOTE: Ensure GROQ_API_KEY is set in your Vercel Project Settings

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Setup CORS
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        # Handle Preflight
        if self.command == 'OPTIONS':
            return

        # 2. Parse Request
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode('utf-8'))
            user_message = body.get('message', '')
        except Exception as e:
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode('utf-8'))
            return

        # 3. Define Agents
        # We re-initialize agents per request to keep the function stateless
        try:
            tech_agent = Agent(
                name="tech",
                instructions="Debug code.",
                model="groq:openai/gpt-oss-20b"
            )

            data_agent = Agent(
                name="data",
                instructions="Explain data analysis concepts.",
                model="groq:openai/gpt-oss-20b"
            )

            docs_agent = Agent(
                name="docs",
                instructions="Write documentation and summaries.",
                model="groq:openai/gpt-oss-20b"
            )

            # 4. Define Team with Routing Tags
            team = Team(
                model="groq:openai/gpt-oss-20b",
                instructions=[
                    "Choose the best agent and let them answer.",
                    "IMPORTANT: You must start your response with a tag identifying the agent used.",
                    "Use [[TECH]] for the tech/debugging agent.",
                    "Use [[DATA]] for the data analysis agent.",
                    "Use [[DOCS]] for the documentation agent.",
                    "If you answer directly, use [[TEAM]].",
                    "Example: '[[TECH]] Here is the fix for your code...'"
                ],
                members=[tech_agent, data_agent, docs_agent]
            )

            # 5. Run Team
            response = team.run(user_message)
            
            # Extract content safely
            ai_content = response.content if hasattr(response, 'content') else str(response)

            # 6. Send Response
            response_data = {
                "choices": [
                    {
                        "message": {
                            "content": ai_content
                        }
                    }
                ]
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            error_response = {
                "error": str(e),
                "details": "Check server logs or API Keys"
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
