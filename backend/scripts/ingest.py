from __future__ import annotations
import argparse
import hashlib
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Tuple
from uuid import uuid4

from tqdm import tqdm

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds backend/


# Our RAG pipeline singleton
from app.rag import rag
from app.settings import settings

# -------- utils --------

def iter_files(root: Path, patterns: List[str]) -> Iterable[Path]:
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and not p.name.startswith("."):
                yield p

def read_text(path: Path) -> str:
    # Simple UTF-8 read; ignore errors rather than crashing
    return path.read_text(encoding="utf-8", errors="ignore")

def read_csv_medical_qa(path: Path, question_col: str = "input", answer_col: str = "output") -> List[Tuple[str, str]]:
    """
    Read a CSV file with medical Q&A pairs.
    Returns list of (question, answer) tuples.
    """
    try:
        df = pd.read_csv(path)
        
        # Try to find question/answer columns (case insensitive)
        cols = df.columns.str.lower()
        
        # Look for question column (input)
        q_col = None
        for col in df.columns:
            if col.lower() in ['input', 'question', 'q', 'query', 'patient_question', 'user_question']:
                q_col = col
                break
        
        # Look for answer column (output)
        a_col = None
        for col in df.columns:
            if col.lower() in ['output', 'answer', 'a', 'response', 'doctor_answer', 'doctor_response', 'reply']:
                a_col = col
                break
        
        if q_col is None or a_col is None:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Could not find input/output columns in {path}")
        
        print(f"Using columns: Input='{q_col}', Output='{a_col}'")
        
        qa_pairs = []
        for _, row in df.iterrows():
            question = str(row[q_col]).strip()
            answer = str(row[a_col]).strip()
            
            # Skip empty or NaN entries
            if question and answer and question != 'nan' and answer != 'nan':
                qa_pairs.append((question, answer))
        
        return qa_pairs
    
    except Exception as e:
        print(f"Error reading CSV {path}: {e}")
        return []

def approximate_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Very simple chunker by character length (≈ tokens).
    Keeps small overlaps to preserve context across chunks.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        # extend to the next sentence boundary if possible (light heuristic)
        while end < n and not text[end - 1] in ".!?\n" and (end - start) < chunk_size + 200:
            end += 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def stable_id(path: Path, idx: int) -> str:
    h = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{idx}-{h}"

def process_csv_file(csv_path: Path, chunk_size: int, overlap: int, deterministic_ids: bool) -> Tuple[List[str], List[str], List[dict]]:
    """Process a CSV file and return ids, docs, metadata for ingestion."""
    qa_pairs = read_csv_medical_qa(csv_path)
    if not qa_pairs:
        return [], [], []
    
    print(f"Found {len(qa_pairs)} Q&A pairs. Processing...")
    
    ids = []
    docs = []
    metas = []
    
    # Process in smaller batches to show progress
    batch_size = 100
    for batch_start in range(0, len(qa_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(qa_pairs))
        print(f"Processing pairs {batch_start+1}-{batch_end} of {len(qa_pairs)}...")
        
        for i in range(batch_start, batch_end):
            question, answer = qa_pairs[i]
            
            # Create a combined document from Q&A
            # Format: "Question: ... Answer: ..."
            combined_text = f"Question: {question}\n\nAnswer: {answer}"
            
            # Check if we need to chunk long answers
            if len(combined_text) > chunk_size:
                chunks = approximate_chunks(combined_text, chunk_size, overlap)
            else:
                chunks = [combined_text]
            
            # Add each chunk
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = stable_id(csv_path, i * 100 + chunk_idx) if deterministic_ids else f"{csv_path.stem}-qa{i}-{chunk_idx}-{uuid4().hex[:8]}"
                
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({
                    "title": f"Medical Q&A {i+1}" + (f" (Part {chunk_idx+1})" if len(chunks) > 1 else ""),
                    "source_path": str(csv_path.name),
                    "question": question,
                    "answer": answer,
                    "qa_pair_index": i,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "content_type": "medical_qa"
                })
    
    return ids, docs, metas

def process_text_file(file_path: Path, chunk_size: int, overlap: int, deterministic_ids: bool) -> Tuple[List[str], List[str], List[dict]]:
    """Process a text/markdown file and return ids, docs, metadata for ingestion."""
    text = read_text(file_path)
    chunks = approximate_chunks(text, chunk_size, overlap)
    if not chunks:
        return [], [], []
    
    ids = []
    docs = []
    metas = []
    
    for i, chunk in enumerate(chunks):
        doc_id = stable_id(file_path, i) if deterministic_ids else f"{file_path.stem}-{i}-{uuid4().hex[:8]}"
        
        ids.append(doc_id)
        docs.append(chunk)
        metas.append({
            "title": file_path.stem.replace("_", " ").strip(),
            "source_path": str(file_path.name),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "content_type": "document"
        })
    
    return ids, docs, metas

# -------- main --------

def main():
    parser = argparse.ArgumentParser(description="Ingest .txt/.md/.csv files into Chroma collection.")
    parser.add_argument("--path", type=str, default=str(Path(__file__).resolve().parents[2] / "data"),
                        help="Root folder containing documents (default: project_root/data)")
    parser.add_argument("--glob", type=str, default="*.txt,*.md,*.csv", help="Comma-separated glob patterns (default: *.txt,*.md,*.csv)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Approximate characters per chunk")
    parser.add_argument("--overlap", type=int, default=120, help="Characters to overlap between chunks")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection before ingesting")
    parser.add_argument("--deterministic-ids", action="store_true",
                        help="Use stable IDs based on file path (avoids duplicate adds on re-run)")

    args = parser.parse_args()
    data_root = Path(args.path).resolve()
    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]

    if not data_root.exists():
        raise SystemExit(f"Data path not found: {data_root}")

    if args.reset:
        # Wipe the collection, then recreate
        try:
            rag.client.delete_collection(settings.collection_name)
        except Exception:
            pass
        rag.collection = rag.client.get_or_create_collection(settings.collection_name)
        print(f"Reset collection: {settings.collection_name}")

    files = list(iter_files(data_root, patterns))
    if not files:
        print(f"No files found in {data_root} matching {patterns}")
        return

    total_chunks = 0
    csv_files = []
    text_files = []
    
    # Separate CSV and text files
    for f in files:
        if f.suffix.lower() == '.csv':
            csv_files.append(f)
        else:
            text_files.append(f)
    
    # Process CSV files (medical Q&A)
    if csv_files:
        print(f"\nProcessing {len(csv_files)} CSV file(s)...")
        for f in tqdm(csv_files, desc="Processing CSV"):
            ids, docs, metas = process_csv_file(f, args.chunk_size, args.overlap, args.deterministic_ids)
            if ids:
                # Process in batches to avoid memory issues
                batch_size = 50
                print(f"Ingesting {len(ids)} chunks in batches of {batch_size}...")
                
                for i in range(0, len(ids), batch_size):
                    batch_end = min(i + batch_size, len(ids))
                    batch_ids = ids[i:batch_end]
                    batch_docs = docs[i:batch_end]
                    batch_metas = metas[i:batch_end]
                    
                    print(f"  Ingesting batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
                    rag.add_texts(batch_ids, batch_docs, batch_metas)
                
                total_chunks += len(ids)
                print(f"  {f.name}: {len(ids)} Q&A entries ingested")
    
    # Process text files
    if text_files:
        print(f"\nProcessing {len(text_files)} text file(s)...")
        for f in tqdm(text_files, desc="Processing text"):
            ids, docs, metas = process_text_file(f, args.chunk_size, args.overlap, args.deterministic_ids)
            if ids:
                rag.add_texts(ids, docs, metas)
                total_chunks += len(ids)

    print(f"\nIngested {len(files)} files, {total_chunks} chunks into '{settings.collection_name}'.")
    # show where Chroma is storing data
    from app.rag import _resolve_chroma_path
    print(f"Chroma store at: {_resolve_chroma_path()}")
    print("Tip: re-run with --deterministic-ids to prevent duplicates on repeated ingests.")

if __name__ == "__main__":
    main()