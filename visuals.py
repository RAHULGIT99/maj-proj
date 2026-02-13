import io
import base64
import json
import random
import asyncio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional
from fastapi import HTTPException, Header
from pydantic import BaseModel
from langchain_groq import ChatGroq
from config import GROQ_API_KEYS
import rag  # Importing rag to call ask directly


# --- Helper: Format conversation history ---
def _format_history(history: Optional[List[dict]] = None) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-10:]:
        role = msg.get("role", "").upper()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"


# --- Data Models ---
class VisualRequest(BaseModel):
    query: str
    index: str  # The index name from the frontend
    history: Optional[List[dict]] = []

class VisualResponse(BaseModel):
    response_type: str  # "viz" or "chat"
    message: str
    task: Optional[str] = None
    visualization_type: Optional[str] = None
    images: Optional[List[str]] = None  # List of base64 strings

# --- Helper: Code Executor ---
def execute_generated_code(code_str: str) -> List[str]:
    """
    Executes Matplotlib code and returns a list of base64 encoded images.
    """
    # Create a safe execution context
    exec_globals = {"plt": plt}
    images_base64 = []
    
    try:
        # Execute the generated code
        exec(code_str, exec_globals)
        
        # Check if any plot was created
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            images_base64.append(img_str)
            plt.close('all') # Important: Clear the plot for next request
            
    except Exception as e:
        print(f"Code execution error: {e}")
        return []
        
    return images_base64

# --- Visualization Logic ---
async def create_visuals(data: VisualRequest, authorization: Optional[str] = Header(None)) -> VisualResponse:
    query = data.query
    index_name = data.index

    # ── Step 0: Classify intent — does the user want a chart or just chatting? ──
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ API keys not configured.")

    history_block = _format_history(data.history)
    classify_prompt = (
        "You are a smart intent classifier. Based on the conversation history and the latest user message, "
        "decide if the user wants to GENERATE or MODIFY a chart/visualization/graph, OR if they are just asking a general question or chatting.\n\n"
        "Reply with EXACTLY one word:\n"
        "- 'viz' if they want to create, modify, or regenerate a chart/graph/visualization\n"
        "- 'chat' if they are asking a question, greeting, or anything else\n\n"
        f"{history_block}"
        f"USER MESSAGE: {query}\n\n"
        "INTENT:"
    )

    key = random.choice(GROQ_API_KEYS)
    try:
        llm = ChatGroq(temperature=0.0, groq_api_key=key, model_name="llama-3.3-70b-versatile")
        classify_resp = await llm.ainvoke(classify_prompt)
        intent = classify_resp.content.strip().lower().split()[0]  # first word only
    except Exception:
        intent = "viz"  # fallback: assume they want a chart

    print(f"Viz intent classified as: {intent}")

    # ── If intent is chat, use RAG /ask with history and return text ──
    if intent != "viz":
        try:
            chat_history = [
                rag.ChatMessage(role=m.get("role", ""), text=m.get("text", ""))
                for m in (data.history or [])
            ]
            ask_request = rag.AskRequest(index_name=index_name, question=query, history=chat_history)
            ask_response = await rag.ask(ask_request, authorization)
            return VisualResponse(
                response_type="chat",
                message=ask_response.answer,
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"Chat fallback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Step 1: Get RAG answer ──
    try:
        ask_request = rag.AskRequest(index_name=index_name, question=query)
        ask_response = await rag.ask(ask_request, authorization)
        rag_answer = ask_response.answer
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error calling rag.ask: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data for visualization: {str(e)}")

    if not rag_answer or rag_answer == "This information is not available in the indexed documents.":
         pass # Proceeding as original

    # ── Step 2: Generate Python Code using Groq ──
    history_block = _format_history(data.history)
    prompt = (
        "You are a Data Visualization Code Generator.\n\n"
        "STRICT RULES — FOLLOW EVERY SINGLE ONE:\n"
        "1. Use ONLY 'matplotlib.pyplot' (imported as plt). Do NOT call plt.show().\n"
        "2. Extract data values EXACTLY as they appear in the DATA section below. "
        "DO NOT invent, estimate, or hallucinate any numbers. "
        "If DATA says 'Market Cap: 3.94T', use 3.94 (in Trillions). If it says '226.5B', use 226.5 (in Billions).\n"
        "3. Keep the UNIT consistent across all values. Convert all to the same unit "
        "(e.g., all in Trillions or all in Billions). Label the axis with the unit you chose (e.g., 'Market Cap (T)' or 'Revenue (B)').\n"
        "4. Use plt.figure(figsize=(12, 7)) for good sizing.\n"
        "5. Use plt.xticks(rotation=45, ha='right') for readable x-labels. Use plt.tight_layout().\n"
        "6. For bar charts, use distinct colors or a colormap for visual clarity.\n"
        "7. If the user's follow-up request asks to change the chart (e.g., scale, style, type), "
        "apply the change to the SAME data from the DATA section — do NOT re-extract or guess new numbers.\n"
        "8. If the user asks for a specific y-axis scale (e.g., '100, 200, 300'), set plt.yticks accordingly "
        "and make sure the data values match the unit implied by that scale.\n\n"
        "Return ONLY a valid JSON object with this structure:\n"
        "{\n"
        "  \"task\": \"Short description of what is being visualized\",\n"
        "  \"visualization_type\": \"Type of chart (e.g., Bar Chart, Pie Chart)\",\n"
        "  \"code\": \"The python code string (use \\n for newlines)\"\n"
        "}\n\n"
        f"{history_block}"
        f"DATA:\n{rag_answer}\n\n"
        f"USER REQUEST: {query}\n"
        "JSON OUTPUT:"
    )
    

    model_name = "llama-3.3-70b-versatile"
    # Ensure GROQ_API_KEYS is available
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ API keys not configured.")
        
    key = random.choice(GROQ_API_KEYS)
    
    try:
        llm = ChatGroq(temperature=0.0, groq_api_key=key, model_name=model_name)
        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        
        # Clean up Markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
             content = content.split("```")[1].split("```")[0].strip()
             
        parsed_json = json.loads(content)
        
        task = parsed_json.get("task", "Visualization")
        viz_type = parsed_json.get("visualization_type", "Chart")
        code = parsed_json.get("code", "")
        
        print(f"Generated Code for {viz_type}:\n{code}")

    except Exception as e:
        print(f"LLM Generation Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate visualization code.")

    # 3. Execute Code and Get Base64 Images
    images = await asyncio.to_thread(execute_generated_code, code)
    
    if not images:
        # Fallback: If no image generated, it might be a code error
        raise HTTPException(status_code=500, detail="Code executed but no image was produced.")

    return VisualResponse(
        response_type="viz",
        message=f"Generated {viz_type}: {task}",
        task=task,
        visualization_type=viz_type,
        images=images
    )
