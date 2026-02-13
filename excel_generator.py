import io
import base64
import json
import random
from typing import Optional
from fastapi import HTTPException, Header
from pydantic import BaseModel
import pandas as pd
from langchain_groq import ChatGroq
from config import GROQ_API_KEYS
import rag


# --- Helper: Format conversation history ---
def _format_history(history: Optional[list] = None) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-10:]:
        role = msg.get("role", "").upper()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"


# --- Data Models ---
class ExcelRequest(BaseModel):
    query: str
    index: str  # The index name from the frontend
    history: Optional[list] = []


class ExcelResponse(BaseModel):
    message: str
    response_type: str  # "excel" or "chat"
    filename: Optional[str] = None
    file_base64: Optional[str] = None  # Base64 encoded .xlsx bytes


# --- Helper: Convert structured JSON to Base64 Excel ---
def build_excel_base64(table_data: list[dict]) -> str:
    """
    Takes a list of dicts (rows), creates a pandas DataFrame,
    writes it to an in-memory Excel buffer, returns Base64 string.
    """
    df = pd.DataFrame(table_data)

    # Clean column headers: strip markdown bold markers like **col**
    df.columns = [col.strip().strip("*") for col in df.columns]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# --- Main Logic ---
async def create_excel(
    data: ExcelRequest, authorization: Optional[str] = Header(None)
) -> ExcelResponse:
    query = data.query
    index_name = data.index

    # ── Step 0: Classify intent — does the user want an Excel file or just chatting? ──
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ API keys not configured.")

    history_block = _format_history(data.history)
    classify_prompt = (
        "You are a smart intent classifier. Based on the conversation history and the latest user message, "
        "decide if the user wants to GENERATE or MODIFY an Excel/spreadsheet/table, OR if they are just asking a general question or chatting.\n\n"
        "Reply with EXACTLY one word:\n"
        "- 'excel' if they want to create, modify, restructure, or download a spreadsheet/table/Excel\n"
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
        intent = "excel"  # fallback: assume they want Excel

    print(f"Intent classified as: {intent}")

    # ── If intent is chat, use RAG /ask with history and return text ──
    if intent != "excel":
        try:
            # Convert raw dicts to ChatMessage objects for rag.AskRequest
            chat_history = [
                rag.ChatMessage(role=m.get("role", ""), text=m.get("text", ""))
                for m in (data.history or [])
            ]
            ask_request = rag.AskRequest(index_name=index_name, question=query, history=chat_history)
            ask_response = await rag.ask(ask_request, authorization)
            return ExcelResponse(
                message=ask_response.answer,
                response_type="chat",
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"Chat fallback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Step 1: Get RAG answer (same pattern as visuals.py) ──
    try:
        ask_request = rag.AskRequest(index_name=index_name, question=query)
        ask_response = await rag.ask(ask_request, authorization)
        rag_answer = ask_response.answer
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calling rag.ask: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve data for Excel: {str(e)}",
        )

    if not rag_answer:
        raise HTTPException(status_code=400, detail="No data found to export.")

    # ── Step 2: Ask LLM to structure the text into JSON rows ──
    history_block = _format_history(data.history)
    prompt = (
        "You are a Data Structuring Assistant.\n"
        "Convert the text data below into a JSON array of objects suitable for an Excel spreadsheet.\n\n"
        "RULES:\n"
        "- Each object in the array represents ONE ROW.\n"
        "- The keys of every object are the COLUMN HEADERS (use short, clean names).\n"
        "- Remove any markdown formatting (**, ##, etc.) from values.\n"
        "- If the data compares entities (e.g. companies), each entity should be its own row.\n"
        "- If the data is a list of metrics, each metric should be its own row.\n"
        "- Keep numbers as numbers, not strings.\n"
        "- Use the conversation history (if any) to understand follow-up requests like restructuring or reformatting.\n"
        "- Return ONLY the raw JSON array. No explanation, no markdown code fences.\n\n"
        f"{history_block}"
        f"DATA:\n{rag_answer}\n\n"
        f"USER REQUEST: {query}\n\n"
        "JSON ARRAY:"
    )

    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ API keys not configured.")

    key = random.choice(GROQ_API_KEYS)

    try:
        llm = ChatGroq(
            temperature=0.0,
            groq_api_key=key,
            model_name="llama-3.3-70b-versatile",
        )
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if LLM wraps them anyway
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        table_data: list[dict] = json.loads(content)

        if not isinstance(table_data, list) or len(table_data) == 0:
            raise ValueError("LLM returned empty or non-list JSON.")

        print(f"Structured {len(table_data)} rows for Excel.")

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}\nRaw content:\n{content}")
        raise HTTPException(
            status_code=500,
            detail="Failed to structure data into table format.",
        )
    except Exception as e:
        print(f"LLM structuring error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate structured data for Excel.",
        )

    # ── Step 3: Build Excel in memory and encode to Base64 ──
    try:
        file_base64 = build_excel_base64(table_data)
    except Exception as e:
        print(f"Excel generation error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate Excel file."
        )

    return ExcelResponse(
        message=f"Excel generated with {len(table_data)} rows.",
        response_type="excel",
        filename="export.xlsx",
        file_base64=file_base64,
    )
