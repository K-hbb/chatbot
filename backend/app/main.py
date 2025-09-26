# Complete fix for your backend/app/main.py - Replace the relevant sections

import google.generativeai as genai
import os
from typing import List
import logging 
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

# --- IMPORTANT: Force direct Gemini API usage ---
# Clear any existing Google Cloud environment variables that might interfere
for env_var in ['GOOGLE_APPLICATION_CREDENTIALS', 'GCLOUD_PROJECT', 'GOOGLE_CLOUD_PROJECT']:
    if env_var in os.environ:
        print(f"WARNING: Clearing {env_var} environment variable to avoid Vertex AI conflicts")
        del os.environ[env_var]

# Configure Gemini API directly
if not settings.gemini_api_key:
    print("WARNING: No Gemini API key found. /chat endpoint will not work.")
    genai_configured = False
else:
    try:
        # Configure ONLY the Gemini API (not Vertex AI)
        genai.configure(api_key=settings.gemini_api_key)
        
        # Test the configuration immediately
        print("Testing Gemini API configuration...")
        test_models = genai.list_models()
        available_models = []
        for model in test_models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        print(f"Available Gemini API models: {available_models}")
        genai_configured = True
        
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
        genai_configured = False

model = None

def _get_model():
    global model
    if model is None:
        if not genai_configured:
            raise RuntimeError("GEMINI_API_KEY not set or invalid")
        
        # Try different model names in order of preference
        model_names_to_try = [
            "gemini-1.5-flash-8b",  # Another free option
            "gemini-2.0-flash",  # If you have access
            "gemini-2.5-flash"   # Latest if available
        ]
        
        last_error = None
        for model_name in model_names_to_try:
            try:
                print(f"DEBUG: Trying model: {model_name}")
                
                # Create model WITHOUT "models/" prefix
                test_model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple prompt
                test_response = test_model.generate_content(
                    "Say 'Model test successful'", 
                    stream=False
                )
                
                if hasattr(test_response, 'text') and test_response.text:
                    print(f"DEBUG: Model {model_name} works! Response: {test_response.text}")
                    model = test_model
                    return model
                else:
                    print(f"DEBUG: Model {model_name} created but no text in response")
                    
            except Exception as e:
                print(f"DEBUG: Model {model_name} failed: {e}")
                last_error = e
                continue
        
        # If we get here, all models failed
        raise RuntimeError(f"Could not create any Gemini model. Last error: {last_error}")
    
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

# --- System prompt ---
SYSTEM_RULES = """You are a caring, friendly medical information assistant.

GREETING (default for casual “hi/hello” messages)
- English: "Hi! I’m your health info assistant. What’s bothering you today? Share your main symptom, when it started, and anything you’ve tried. If you have severe symptoms (trouble breathing, chest pain, heavy bleeding, confusion), seek urgent care right now."
- French:  "Bonjour ! Je suis votre assistant d’information santé. Qu’est-ce qui vous gêne aujourd’hui ? Indiquez le symptôme principal, depuis quand il a commencé et ce que vous avez déjà essayé. Si vous avez des symptômes graves (difficulté à respirer, douleur thoracique, saignement abondant, confusion), consultez en urgence immédiatement."
- Arabic:  "مرحبًا! أنا مساعدك لمعلومات الصحة. ما الذي يزعجك اليوم؟ اذكر العرض الرئيسي، متى بدأ، وما الذي جرّبته. إذا كانت لديك أعراض شديدة (صعوبة في التنفّس، ألم في الصدر، نزيف غزير، ارتباك)، فاطلب رعاية طبية عاجلة فورًا."
- Use the user’s language automatically. If unclear, default to English.

SAFETY & SCOPE
- Provide general, educational information only. Do NOT diagnose, prescribe, or replace professional advice.
- Use ONLY the provided context from medical Q&A pairs. If information is missing or uncertain, say you don’t know and suggest consulting a clinician.
- Never fabricate citations or facts. If a source is required, reference only the provided medical consultation examples by name (no external links).
- If the user reports emergency red flags (e.g., trouble breathing, chest pain, severe bleeding, confusion, signs of stroke, anaphylaxis, severe dehydration, suicidal intent), immediately advise seeking emergency care now and stop giving non-urgent advice.

INTAKE & TRIAGE (when user describes a problem)
- Ask only what’s necessary to move forward. Prefer concise checklists.
- Useful details: main symptom & onset, fever (max °C/°F), pain 0–10, key associated symptoms, age & sex, relevant conditions/meds/allergies, pregnancy (if applicable), what’s been tried, worsening pattern.
- Offer clear next steps (self-care options, when to seek in-person care, what to monitor). Avoid definitive diagnoses.

RESPONSE STYLE
- Warm, empathetic, and professional.
- Use clear, accessible language and short paragraphs.
- Structure with brief headers or bullets when helpful.
- Be concise; prioritize actionable guidance.

LANGUAGE & TONE
- Mirror the user’s language (EN/FR/AR supported here). Keep medical terms plain; define any unavoidable jargon.

BOUNDARIES & PRIVACY
- Do not request full personal identifiers. Avoid collecting unnecessary sensitive data.
- Do not provide instructions for illegal, unsafe, or hazardous activities.

"""


def build_prompt(question: str, contexts: List[str], metas: List[dict]) -> str:
    lines = [SYSTEM_RULES, "\n# Context (retrieved medical information)\n"]
    for i, (c, m) in enumerate(zip(contexts, metas), 1):
        title = (m or {}).get("title", f"Medical Source {i}")
        src = (m or {}).get("source_path", "medical_knowledge")
        lines.append(f"## [{i}] {title} (source: {src})\n{c.strip()}\n")
    lines.append("\n# Task\nAnswer the user question using the context above. Be helpful and direct.\n")
    lines.append(f"User question: {question}\n")
    return "\n".join(lines)

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint with robust Gemini API handling.
    """
    def generate():
        try:
            print(f"DEBUG: Starting chat request for: {req.question}")
            
            # Safety check
            is_emerg, disclaimer = screen_safety(req.question)
            print(f"DEBUG: Safety check - Emergency: {is_emerg}")
            
            # Send safety info first
            yield f"data: {json.dumps({'type': 'safety', 'emergency': is_emerg, 'disclaimer': disclaimer})}\n\n"
            
            # Retrieve documents
            try:
                print("DEBUG: Starting document retrieval...")
                ids, docs, metas, sims = rag.query(req.question, k=5)
                print(f"DEBUG: Retrieved {len(ids or [])} documents")
            except Exception as e:
                print(f"DEBUG: Retrieval failed: {e}")
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
            
            sources_data = [src.model_dump() for src in srcs]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"
            print(f"DEBUG: Sent {len(srcs)} sources to frontend")

            # Build prompt
            prompt = build_prompt(req.question, contexts, context_metas)
            print(f"DEBUG: Built prompt length: {len(prompt)}")

            # Get model
            try:
                mdl = _get_model()
                print(f"DEBUG: Got model successfully")
                print(f"DEBUG: Model details: {mdl}")
            except Exception as e:
                print(f"DEBUG: Model creation failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            # Generate response
            try:
                print("DEBUG: Starting generation...")
                
                # Try streaming first
                try:
                    response = mdl.generate_content(
                        prompt, 
                        stream=True,
                        generation_config={
                            'temperature': 0.7,
                            'max_output_tokens': 1024,
                        }
                    )
                    
                    has_content = False
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            has_content = True
                            yield f"data: {json.dumps({'type': 'text', 'data': chunk.text})}\n\n"
                        elif hasattr(chunk, 'parts'):
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    has_content = True
                                    yield f"data: {json.dumps({'type': 'text', 'data': part.text})}\n\n"
                    
                    if not has_content:
                        raise Exception("No content received from streaming")
                    
                except Exception as streaming_error:
                    print(f"DEBUG: Streaming failed: {streaming_error}")
                    # Try non-streaming fallback
                    response = mdl.generate_content(
                        prompt, 
                        stream=False,
                        generation_config={
                            'temperature': 0.7,
                            'max_output_tokens': 1024,
                        }
                    )
                    
                    if hasattr(response, 'text') and response.text:
                        yield f"data: {json.dumps({'type': 'text', 'data': response.text})}\n\n"
                    else:
                        raise Exception("No content from non-streaming either")
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                print("DEBUG: Generation completed successfully")
                
            except Exception as e:
                print(f"DEBUG: Generation failed: {e}")
                # Send context-based fallback
                if contexts:
                    fallback_text = "I apologize for the technical difficulty. Based on the medical information I found:\n\n"
                    for i, (doc, meta) in enumerate(zip(contexts[:2], context_metas[:2]), 1):
                        title = (meta or {}).get("title", f"Medical Source {i}")
                        snippet = doc[:200].strip()
                        fallback_text += f"**{title}:** {snippet}...\n\n"
                    fallback_text += "For personalized advice, please consult a healthcare professional."
                else:
                    fallback_text = "I'm experiencing technical difficulties. Please try rephrasing your question or contact a healthcare professional for urgent concerns."
                
                yield f"data: {json.dumps({'type': 'text', 'data': fallback_text})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print(f"DEBUG: Top-level exception: {type(e).__name__}: {e}")
            error_msg = "I'm experiencing technical difficulties. Please try again or contact a healthcare professional for urgent medical questions."
            yield f"data: {json.dumps({'type': 'text', 'data': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Non-streaming fallback endpoint
@app.post("/chat-sync", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming fallback endpoint"""
    try:
        # Safety
        is_emerg, disclaimer = screen_safety(req.question)
        
        # Retrieve
        ids, docs, metas, sims = rag.query(req.question, k=5)
        contexts = docs or []
        context_metas = metas or []
        
        # Build prompt
        prompt = build_prompt(req.question, contexts, context_metas)
        
        # Get model and generate
        mdl = _get_model()
        result = mdl.generate_content(
            prompt,
            stream=False,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1024,
            }
        )
        
        text = result.text if hasattr(result, 'text') and result.text else "Unable to generate response"
        
        # Build sources
        srcs = []
        for _id, mt, s in zip(ids or [], metas or [], sims or []):
            sid = _id or (mt or {}).get("source_path") or "unknown"
            srcs.append(Source(id=str(sid), score=float(s), title=(mt or {}).get("title"), metadata=dict(mt or {})))
        
        return ChatResponse(text=text, sources=srcs, emergency=is_emerg, disclaimer=disclaimer)
        
    except Exception as e:
        logging.getLogger(__name__).exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")