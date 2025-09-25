from __future__ import annotations
from typing import List
import logging 
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import google.generativeai as genai

from .settings import settings
from .rag import rag
from .models import SearchRequest, SearchResponse, SearchHit, ChatRequest, ChatResponse, Source
from .safety import screen_safety

app = FastAPI(title=settings.app_name)

# --- CORS for local dev ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Gemini once on startup ---
if not settings.gemini_api_key:
    # We'll still start; /chat will raise a helpful 500 if key missing
    genai_configured = False
else:
    genai.configure(api_key=settings.gemini_api_key)
    genai_configured = True

model = None
def _get_model():
    global model
    if model is None:
        if not genai_configured:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        
        # Use the model that we know works from testing
        model_name = "models/gemini-2.0-flash"
        print(f"DEBUG: Creating Gemini model with name: {model_name}")
        
        try:
            model = genai.GenerativeModel(model_name)
            print(f"DEBUG: Successfully created model: {model_name}")
        except Exception as e:
            print(f"DEBUG: Failed to create model {model_name}: {e}")
            raise RuntimeError(f"Could not create Gemini model: {e}")
    
    return model

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name, "model": settings.embedding_model}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    ids, docs, metas, sims = rag.query(req.query, k=req.k)
    hits = []
    for _id, d, m, s in zip(ids, docs, metas, sims):
        title = (m or {}).get("title")
        snippet = (d or "")[:200].replace("\n", " ").strip()
        hits.append(
            SearchHit(
                doc_id=_id or (m or {}).get("source_path", "unknown"),
                title=title,
                snippet=snippet,
                score=float(s),
                metadata=m or {}
            )
        )
    return SearchResponse(hits=hits)

# --- Prompt builder ---
SYSTEM_RULES = """You are a cautious, helpful medical information assistant.
- Provide general, educational information. Do NOT diagnose, prescribe, or replace professional advice.
- Use ONLY the provided context from medical Q&A pairs; if missing, say you don't know and suggest consulting a clinician.
- Keep answers concise and structured. Include clear, actionable next steps when appropriate.
- Never invent citations. Cite sources based on the Q&A context chunks provided.
- When referencing Q&A pairs, mention that the information comes from medical consultation examples.
"""

def build_prompt(question: str, contexts: List[str], metas: List[dict]) -> str:
    lines = [SYSTEM_RULES, "\n# Context (top retrieved chunks)\n"]
    for i, (c, m) in enumerate(zip(contexts, metas), 1):
        title = (m or {}).get("title", f"Doc {i}")
        src = (m or {}).get("source_path", "unknown")
        lines.append(f"## [{i}] {title} (source: {src})\n{c.strip()}\n")
    lines.append("\n# Task\nAnswer the user question using only the context above. If unsure, say so clearly.\n")
    lines.append(f"User question: {question}\n")
    lines.append("Return a short answer (5-8 sentences max) followed by a 'Sources:' list like [1], [2] referencing the context numbers.\n")
    return "\n".join(lines)

# --- Streaming Chat Endpoint ---
# Replace the streaming chat endpoint in your backend/app/main.py with this version:

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint that returns Server-Sent Events.
    """
    def generate():
        try:
            print(f"DEBUG: Starting chat request for: {req.question}")
            
            # Safety check
            is_emerg, disclaimer = screen_safety(req.question)
            print(f"DEBUG: Safety check - Emergency: {is_emerg}, Disclaimer length: {len(disclaimer)}")
            
            # Send safety info first
            yield f"data: {json.dumps({'type': 'safety', 'emergency': is_emerg, 'disclaimer': disclaimer})}\n\n"
            
            # Retrieve documents
            try:
                print("DEBUG: Starting document retrieval...")
                ids, docs, metas, sims = rag.query(req.question, k=5)
                print(f"DEBUG: Retrieved {len(ids or [])} documents")
                print(f"DEBUG: First doc preview: {(docs[0][:100] if docs and docs[0] else 'No docs')}")
            except Exception as e:
                print(f"DEBUG: Retrieval failed: {e}")
                logging.getLogger(__name__).exception("Retrieval failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'message': f'Retrieval error: {str(e)}'})}\n\n"
                return

            contexts = docs or []
            context_metas = metas or []
            print(f"DEBUG: Using {len(contexts)} contexts for generation")

            # Send sources
            srcs: list[Source] = []
            for _id, mt, s in zip(ids or [], metas or [], sims or []):
                sid = _id or (mt or {}).get("source_path") or "unknown"
                srcs.append(
                    Source(
                        id=str(sid),
                        score=float(s),
                        title=(mt or {}).get("title"),
                        metadata=dict(mt or {})
                    )
                )
            
            # Send sources to frontend
            sources_data = [src.model_dump() for src in srcs]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"
            print(f"DEBUG: Sent {len(srcs)} sources to frontend")

            # Build prompt
            prompt = build_prompt(req.question, contexts, context_metas)
            print(f"DEBUG: Built prompt length: {len(prompt)}")
            print(f"DEBUG: Prompt preview: {prompt[:200]}...")

            # Get model
            try:
                mdl = _get_model()
                print(f"DEBUG: Got model successfully: {mdl}")
            except RuntimeError as e:
                print(f"DEBUG: Model creation failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            # Stream generation
            try:
                print(f"DEBUG: About to call generate_content...")
                print(f"DEBUG: Model details: {mdl}")
                
                # Use Gemini's streaming
                response = mdl.generate_content(prompt, stream=True)
                print(f"DEBUG: generate_content call successful, got response object")
                
                has_content = False
                chunk_count = 0
                
                for chunk in response:
                    chunk_count += 1
                    print(f"DEBUG: Processing chunk {chunk_count}")
                    
                    if hasattr(chunk, 'text') and chunk.text:
                        has_content = True
                        chunk_text = chunk.text
                        print(f"DEBUG: Chunk {chunk_count} text length: {len(chunk_text)}")
                        print(f"DEBUG: Chunk {chunk_count} preview: {chunk_text[:50]}...")
                        # Send each text chunk as it arrives
                        yield f"data: {json.dumps({'type': 'text', 'data': chunk_text})}\n\n"
                    else:
                        print(f"DEBUG: Chunk {chunk_count} has no text: {chunk}")
                
                print(f"DEBUG: Streaming complete. Total chunks: {chunk_count}, Had content: {has_content}")
                
                if not has_content:
                    print("DEBUG: No content received from Gemini, sending fallback")
                    fallback_text = "I apologize, but I didn't receive any content from the AI model. Please try again."
                    yield f"data: {json.dumps({'type': 'text', 'data': fallback_text})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                print("DEBUG: Sent completion signal")
                
            except Exception as e:
                print(f"DEBUG: Exception during generation: {type(e).__name__}: {e}")
                logging.getLogger(__name__).exception("Gemini generation failed: %s", e)
                
                # Fallback: send retrieval-only response
                print("DEBUG: Sending retrieval-only fallback")
                tops = []
                for i, (d, m) in enumerate(zip(contexts, context_metas), 1):
                    title = (m or {}).get("title", f"Doc {i}")
                    snippet = (d or "")[:220].replace("\n", " ").strip()
                    tops.append(f"[{i}] {title}: {snippet}")
                    if i >= 2:
                        break
                
                if tops:
                    fallback_text = (
                        "I'm having trouble generating a full answer right now. "
                        "Here are key points from the retrieved context:\n\n" + "\n".join(tops)
                    )
                else:
                    fallback_text = "I'm having trouble generating an answer. Please try rephrasing your question."
                
                yield f"data: {json.dumps({'type': 'text', 'data': fallback_text})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print(f"DEBUG: Top-level exception: {type(e).__name__}: {e}")
            logging.getLogger(__name__).exception("Streaming chat failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred'})}\n\n"

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# --- Non-streaming chat endpoint (fallback) ---
@app.post("/chat-sync", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Non-streaming fallback endpoint for clients that don't support SSE.
    """
    # Safety
    is_emerg, disclaimer = screen_safety(req.question)

    # Retrieve
    try:
        ids, docs, metas, sims = rag.query(req.question, k=5)
    except Exception as e:
        logging.getLogger(__name__).exception("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    contexts = docs or []
    context_metas = metas or []

    # Build prompt
    prompt = build_prompt(req.question, contexts, context_metas)

    # Model
    try:
        mdl = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate with fallback + logging
    text = ""
    try:
        result = mdl.generate_content(prompt)
        text = (result.text or "").strip()
        if not text:
            text = "I couldn't generate a full answer. Please consult a clinician for personalized advice."
    except Exception as e:
        logging.getLogger(__name__).exception("Gemini generation failed: %s", e)
        # Retrieval-only fallback so we still return 200
        tops = []
        for i, (d, m) in enumerate(zip(contexts, context_metas), 1):
            title = (m or {}).get("title", f"Doc {i}")
            snippet = (d or "")[:220].replace("\n", " ").strip()
            tops.append(f"[{i}] {title}: {snippet}")
            if i >= 2:
                break
        text = (
            "I'm having trouble generating a full answer right now. "
            "Here are key points from the retrieved context:\n\n" + "\n".join(tops)
        )

    # Build sources safely (ensure id is a non-empty string)
    srcs: list[Source] = []
    for _id, mt, s in zip(ids or [], metas or [], sims or []):
        sid = _id or (mt or {}).get("source_path") or "unknown"
        srcs.append(
            Source(
                id=str(sid),
                score=float(s),
                title=(mt or {}).get("title"),
                metadata=dict(mt or {})
            )
        )

    return ChatResponse(
        text=text,
        sources=srcs,
        emergency=is_emerg,
        disclaimer=disclaimer
    )