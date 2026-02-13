import os
from typing import Optional
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import auth
import rag
import visuals
import excel_generator

load_dotenv()

app = FastAPI(title="Dynamic Web-to-RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth Routes ---
@app.post("/register")
async def register(payload: auth.RegisterRequest):
    return await auth.register_user(payload)

@app.post("/verify-otp")
async def verify_otp(payload: auth.VerifyOtpRequest):
    return await auth.verify_otp(payload)

@app.post("/login")
async def login(payload: auth.LoginRequest):
    return await auth.login_user(payload)

# --- RAG Routes ---
@app.post("/analyze", response_model=rag.AnalyzeResponse)
async def analyze(request: rag.AnalyzeRequest, authorization: Optional[str] = Header(None)):
    return await rag.analyze(request, authorization)

@app.post("/ask", response_model=rag.AskResponse)
async def ask(request: rag.AskRequest, authorization: Optional[str] = Header(None)):
    return await rag.ask(request, authorization)

# --- Visualization Route ---
@app.post("/visuals", response_model=visuals.VisualResponse)
async def visualize(data: visuals.VisualRequest, authorization: Optional[str] = Header(None)):
    return await visuals.create_visuals(data, authorization)

# --- Excel Export Route ---
@app.post("/excel", response_model=excel_generator.ExcelResponse)
async def export_excel(data: excel_generator.ExcelRequest, authorization: Optional[str] = Header(None)):
    return await excel_generator.create_excel(data, authorization)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
