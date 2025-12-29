from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json

app = FastAPI()

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/login")
def login_handler(creds: LoginRequest):
    try:
        # Robust path finding (works on Vercel and Local)
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
        print(f"ERROR: Could not find file at {json_path}")
        raise HTTPException(status_code=500, detail="User database missing")
    except Exception as e:
        print(f"Login Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
