import os
import uuid
import tempfile
import asyncio
import random
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
from trafilatura import extract as trafilatura_extract
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import httpx
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -------------------------
# CONFIG / KEYS
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in env")

API_KEY = os.getenv("API_KEY")  # optional auth for your endpoints

# Groq keys for failover
GROQ_API_KEYS = [k for k in (os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")) if k]
if not GROQ_API_KEYS:
    raise ValueError("At least one GROQ_API_KEY_x environment variable must be set")

EMBEDDING_API_URL =  "https://rahulbro123-embedding-model.hf.space/get_embeddings"
EMBEDDING_API_BATCH_SIZE = int(os.getenv("EMBEDDING_API_BATCH_SIZE", "32")) 

# Pinecone client (serverless)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# -------------------------
# FASTAPI + MODELS
# -------------------------
app = FastAPI(title="Dynamic Web-to-RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of public URLs to scrape and index.")

class AnalyzeResponse(BaseModel):
    success: bool
    index_name: Optional[str]
    summary: Optional[str]

class AskRequest(BaseModel):
    index_name: str = Field(..., description="Dynamic index name returned by /analyze")
    question: str = Field(..., description="User question to answer from the index")

class AskResponse(BaseModel):
    answer: str

# -------------------------
# REMOTE EMBEDDING CLIENT
# -------------------------
class RemoteEmbeddingClient(Embeddings):
    def __init__(self, api_url: str, batch_size: int = 32):
        self.api_url = api_url
        self.batch_size = batch_size
        self.client = httpx.AsyncClient(timeout=60.0)

    async def _call_embedding_api(self, texts: List[str]):
        response = await self.client.post(self.api_url, json={"texts": texts})
        response.raise_for_status()
        data = response.json()
        if "embeddings" not in data:
            raise ValueError("Invalid embedding response")
        return data["embeddings"]

    async def aembed_documents(self, texts: List[str]):
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            tasks.append(self._call_embedding_api(batch))
        results = await asyncio.gather(*tasks)
        embeddings = []
        for r in results:
            embeddings.extend(r)
        return embeddings

    async def aembed_query(self, text: str):
        embs = await self.aembed_documents([text])
        return embs[0]

    # sync wrappers (some libs may call sync)
    def embed_documents(self, texts: List[str]):
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str):
        return asyncio.run(self.aembed_query(text))

embeddings = RemoteEmbeddingClient(api_url=EMBEDDING_API_URL, batch_size=EMBEDDING_API_BATCH_SIZE)

# -------------------------
# SCRAPER: static + JS fallback
# -------------------------
def extract_static(url: str, timeout: int = 12):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        html = resp.text
        text = trafilatura_extract(html)
        if text and len(text.strip()) > 200:
            return text
        # fallback to BS
        soup = BeautifulSoup(html, "html.parser")
        raw = soup.get_text("\n", strip=True)
        if raw and len(raw.strip()) > 200:
            return raw
        return None
    except Exception as e:
        print("extract_static error:", e)
        return None

def extract_js(url: str, timeout: int = 30):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout*1000, wait_until="networkidle")
            html = page.content()
            browser.close()
        text = trafilatura_extract(html)
        if text and len(text.strip()) > 200:
            return text
        soup = BeautifulSoup(html, "html.parser")
        raw = soup.get_text("\n", strip=True)
        if raw and len(raw.strip()) > 200:
            return raw
        return None
    except Exception as e:
        print("extract_js error:", e)
        return None

async def fetch_and_combine(urls: List[str]) -> str:
    parts = []
    loop = asyncio.get_running_loop()
    for url in urls:
        # Try static quickly (run sync in threadpool)
        text = await asyncio.to_thread(extract_static, url)
        if not text:
            # Try JS
            text = await asyncio.to_thread(extract_js, url)
        if not text:
            parts.append(f"[Could not extract content from: {url}]")
        else:
            # optional: add small header for origin
            parts.append(f"--- SOURCE: {url} ---\n{text}\n")
    combined = "\n\n".join(parts).strip()
    return combined

# -------------------------
# PINECONE HELPERS (async wrappers)
# -------------------------
async def create_index_if_not_exists_async(index_name: str, dimension: int = 768):
    def _sync():
        names = pinecone_client.list_indexes().names()
        if index_name not in names:
            print(f"Creating index {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"Index {index_name} exists")
    await asyncio.to_thread(_sync)

def _chunk_text_sync(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.split_text(text)
    return docs

async def process_text_and_upsert(index_name: str, full_text: str):
    # 1. Chunk in thread
    chunks = await asyncio.to_thread(_chunk_text_sync, full_text)
    # 2. Embed chunks in parallel via remote API
    print(f"Embedding {len(chunks)} chunks...")
    vectors = await embeddings.aembed_documents(chunks)
    # 3. Prepare upsert
    to_upsert = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        to_upsert.append({
            "id": f"doc_{i}_{uuid.uuid4().hex[:8]}",
            "values": vec,
            "metadata": {"text": chunk}
        })
    # 4. Upsert via thread (pinecone client is sync)
    def _upsert_sync(vectors_batch):
        index = pinecone_client.Index(index_name)
        print(f"Upserting {len(vectors_batch)} vectors to {index_name}")
        # Upsert in batches if needed
        index.upsert(vectors=vectors_batch, batch_size=100)
    await asyncio.to_thread(_upsert_sync, to_upsert)

# -------------------------
# Groq LLM utilities (failover)
# -------------------------
async def generate_summary_with_groq(combined_text: str) -> str:
    print("combined_text is ", combined_text)
    print()
    prompt = (
        "You are an experienced equity research analyst. "
        "Given the combined extracted content below from multiple web pages, produce a concise, "
        "professional, user-friendly analyst summary in about 10 short lines. "
        "Focus on the most important facts, company profile, key metrics or indices mentioned, "
        "and any notable points. Use short sentences and clear language.\n\n"
        "CONTENT:\n"
        + combined_text
        + "\n\nSUMMARY (about 10 lines):"
    )
    shuffled_keys = random.sample(GROQ_API_KEYS, len(GROQ_API_KEYS))
    for i, key in enumerate(shuffled_keys):
        try:
            llm = ChatGroq(temperature=0.0, groq_api_key=key, model_name="llama-3.1-8b-instant")
            resp = await llm.ainvoke(prompt)
            if resp and getattr(resp, "content", None):
                return resp.content.strip()
        except Exception as e:
            print(f"Groq summary error with key #{i+1}: {e}. Trying next key...")
            continue
    return "Summary could not be generated due to an external service error."

async def optimize_text_for_rag(text: str) -> str:
    print("Optimizing text for RAG...")
    prompt = (
        "You are a data processing expert for RAG systems. "
        "Your task is to take the following raw scraped text and rewrite it into a clean, structured, and comprehensive format. "
        "Focus on preserving all factual data, especially financial metrics, numbers, dates, and key entities. "
        "Fix any formatting issues where labels and values are separated by newlines (e.g., change 'Market Cap\n1.34T' to 'Market Cap: 1.34T'). "
        "Remove irrelevant navigation links, ads, or boilerplate footer text. "
        "Ensure the output is dense with information and optimized for semantic search retrieval.\n\n"
        "RAW TEXT:\n"
        + text
        + "\n\nOPTIMIZED TEXT:"
    )
    shuffled_keys = random.sample(GROQ_API_KEYS, len(GROQ_API_KEYS))
    for i, key in enumerate(shuffled_keys):
        try:
            llm = ChatGroq(temperature=0.0, groq_api_key=key, model_name="llama-3.1-8b-instant")
            resp = await llm.ainvoke(prompt)
            if resp and getattr(resp, "content", None):
                return resp.content.strip()
        except Exception as e:
            print(f"Groq optimization error with key #{i+1}: {e}. Trying next key...")
            continue
    # Fallback: return original text if optimization fails
    print("Optimization failed, using original text.")
    return text

async def answer_with_context_groq(question: str, context: str) -> str:
    prompt = (
        "You are a helpful assistant. Answer only based on the CONTEXT below. "
        "If the answer is not present in the context, say: 'I could not find the answer in the provided documents.' "
        "Be concise and formal.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    shuffled_keys = random.sample(GROQ_API_KEYS, len(GROQ_API_KEYS))
    for i, key in enumerate(shuffled_keys):
        try:
            llm = ChatGroq(temperature=0.0, groq_api_key=key, model_name="llama-3.1-8b-instant")
            resp = await llm.ainvoke(prompt)
            if resp and getattr(resp, "content", None):
                return resp.content.strip()
        except Exception as e:
            print(f"Groq answer error with key #{i+1}: {e}. Trying next key...")
            continue
    return "The answer could not be generated due to an external service error."

# -------------------------
# ENDPOINT: /analyze
# -------------------------
@app.post("/analyze", response_model=AnalyzeResponse, tags=["ingest"])
async def analyze(request: AnalyzeRequest, authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    print("Received /analyze request for URLs:", request.urls)

    # 1. Scrape and combine
    combined_text = await fetch_and_combine(request.urls)
    if not combined_text or len(combined_text.strip()) == 0:
        raise HTTPException(status_code=500, detail="Failed to extract content from provided URLs.")

    # 2. Create a dynamic index name
    index_name = f"webindex-{uuid.uuid4().hex[:8]}"
    await create_index_if_not_exists_async(index_name)

    # 3. Optimize text for RAG
    optimized_text = await optimize_text_for_rag(combined_text)

    # 4. Process, chunk, embed and upsert (using optimized text)
    await process_text_and_upsert(index_name, optimized_text)

    # 5. Generate summary via Groq (using original or optimized? Let's use optimized for consistency)
    summary = await generate_summary_with_groq(optimized_text)

    print(f"Analyze complete. Index: {index_name}")

    return AnalyzeResponse(success=True, index_name=index_name, summary=summary)

# -------------------------
# HELPER: search top-k and return combined context
# -------------------------
def _search_index_sync(index_name: str, query_vector: list, top_k: int = 5) -> str:
    try:
        index = pinecone_client.Index(index_name)
        res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = res.get("matches", []) or []
        texts = [m.get("metadata", {}).get("text", "") for m in matches]
        return "\n\n---\n\n".join(texts)
    except Exception as e:
        print("pinecone search error:", e)
        return ""

# -------------------------
# ENDPOINT: /ask
# -------------------------
@app.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(req: AskRequest, authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    if not req.index_name:
        raise HTTPException(status_code=400, detail="index_name required")

    # 1. Embed the question
    query_vec = await embeddings.aembed_query(req.question)

    # 2. Search Pinecone (sync call in thread)
    context = await asyncio.to_thread(_search_index_sync, req.index_name, query_vec, 5)
    if not context.strip():
        return AskResponse(answer="This information is not available in the indexed documents.")

    # 3. Ask Groq with the context and return
    answer = await answer_with_context_groq(req.question, context)
    return AskResponse(answer=answer)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
