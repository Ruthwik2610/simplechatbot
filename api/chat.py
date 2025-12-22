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
                    "You are the Intelligent Routing Orchestrator for ServiceNow Customer Support.",
                    "Your goal is to route user queries to the specific specialist agent best suited to handle them.",
                    
                    "CRITICAL OUTPUT RULE: You MUST start your response with exactly one of the following tags. Do not write any text before the tag.",
                    
                    "1. Use [[TECH]] for technical implementation and debugging.",
                    "   - Context: GlideRecord scripting, Business Rules, Client Scripts, API errors, Flow Designer issues, or instance performance debugging.",
                    
                    "2. Use [[DATA]] for analytics and reporting.",
                    "   - Context: Performance Analytics (PA), Report creation, Dashboard configuration, table statistics, or trend analysis.",
                    
                    "3. Use [[DOCS]] for content generation and explanation.",
                    "   - Context: Writing Knowledge Base (KB) articles, summarizing Release Notes, explaining Standard Operating Procedures (SOPs), or drafting email templates.",
                    
                    "4. Use [[TEAM]] for general queries.",
                    "   - Context: Greetings (e.g., 'Hello'), general questions about the team, or if the request doesn't fit the categories above.",

                    "RESPONSE FORMAT:",
                    "[[TAG]] <Your response content>",
                    
                    "EXAMPLES:",
                    "- User: 'My Business Rule isn't running on update.' -> Response: '[[TECH]] Let's look at your condition and script...'",
                    "- User: 'Create a report for open incidents by assignment group.' -> Response: '[[DATA]] I can explain how to configure that report...'",
                    "- User: 'Draft a KB article for password reset.' -> Response: '[[DOCS]] Here is a draft structure for the article...'"
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
