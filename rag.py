import uuid
import asyncio
import random
from typing import List, Optional
from fastapi import HTTPException, Header
from pydantic import BaseModel, Field
import httpx
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, GROQ_API_KEYS, EMBEDDING_API_URL, EMBEDDING_API_BATCH_SIZE, API_KEY
from scraper import fetch_and_combine

# Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of public URLs to scrape and index.")

class AnalyzeResponse(BaseModel):
    success: bool
    index_name: Optional[str]
    summary: Optional[str]

class ChatMessage(BaseModel):
    role: str
    text: str

class AskRequest(BaseModel):
    index_name: str = Field(..., description="Dynamic index name returned by /analyze")
    question: str = Field(..., description="User question to answer from the index")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation messages for context")

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
# PINECONE HELPERS (async wrappers)
# -------------------------
async def create_index_if_not_exists_async(index_name: str, dimension: int = 768):
    def _sync():
        names = pinecone_client.list_indexes().names()
        if index_name not in names:
            # Feature: if > 5 indexes, delete one then create one
            if len(names) >= 5:
                to_delete = names[0]
                print(f"Index limit reached ({len(names)} >= 5). Deleting old index: {to_delete}")
                try:
                    pinecone_client.delete_index(to_delete)
                except Exception as e:
                    print(f"Error deleting index {to_delete}: {e}")

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
        "professional, user-friendly analyst summary . "
        "Just include everything dont miss anything basically give an overview of the content which was given."
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

async def answer_with_context_groq(question: str, context: str, history: Optional[List] = None) -> str:
    history_block = ""
    if history:
        lines = []
        for msg in history[-10:]:  # Last 10 messages max
            role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
            text = msg.text if hasattr(msg, 'text') else msg.get('text', '')
            lines.append(f"{role.upper()}: {text}")
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        "You are a helpful assistant. Answer only based on the CONTEXT below. "
        "If the answer is not present in the context, say: 'I could not find the answer in the provided documents.' "
        "Use the conversation history to understand follow-up questions. "
        "Be concise and formal.\n\n"
        f"{history_block}"
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
# ENDPOINT: /analyze
# -------------------------
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
    
    print("--------------------------------------------------")
    print("CLEANED TEXT BEFORE VECTOR DB:")
    print(optimized_text)
    print("--------------------------------------------------")

    # 4. Process, chunk, embed and upsert (using optimized text)
    await process_text_and_upsert(index_name, optimized_text)

    # 5. Generate summary via Groq (using original or optimized? Let's use optimized for consistency)
    summary = await generate_summary_with_groq(optimized_text)

    print(f"Analyze complete. Index: {index_name}")

    return AnalyzeResponse(success=True, index_name=index_name, summary=summary)

# -------------------------
# ENDPOINT: /ask
# -------------------------
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

    # 3. Ask Groq with the context and history, then return
    answer = await answer_with_context_groq(req.question, context, req.history)
    return AskResponse(answer=answer)
