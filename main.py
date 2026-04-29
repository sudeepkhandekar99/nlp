import os
import re
import json
import uuid
from pathlib import Path
from typing import Optional, List

import fitz
import faiss
import whisper
import numpy as np
import pytesseract

from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Runtime safety for Mac / local
# -----------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# -----------------------------
# App config
# -----------------------------

app = FastAPI(title="AI Study Assistant Backend")

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


# -----------------------------
# Lazy loaded models
# -----------------------------

embedder = None
llm_tokenizer = None
llm_model = None
whisper_model = None


def get_embedder():
    global embedder

    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return embedder


def get_llm():
    global llm_tokenizer, llm_model

    if llm_tokenizer is None or llm_model is None:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    return llm_tokenizer, llm_model


def get_whisper():
    global whisper_model

    if whisper_model is None:
        whisper_model = whisper.load_model("base")

    return whisper_model


# -----------------------------
# Folder helpers
# -----------------------------

def create_job_folder():
    job_id = str(uuid.uuid4())
    job_folder = OUTPUT_DIR / job_id
    job_folder.mkdir(parents=True, exist_ok=True)

    input_folder = job_folder / "inputs"
    input_folder.mkdir(parents=True, exist_ok=True)

    feature_folder = job_folder / "features"
    feature_folder.mkdir(parents=True, exist_ok=True)

    return job_id, job_folder, input_folder, feature_folder


def get_latest_job_folder():
    folders = [folder for folder in OUTPUT_DIR.iterdir() if folder.is_dir()]

    if not folders:
        raise HTTPException(status_code=404, detail="No processed job folders found.")

    return max(folders, key=lambda folder: folder.stat().st_mtime)


def get_job_folder(job_id: Optional[str] = None):
    if job_id:
        job_folder = OUTPUT_DIR / job_id

        if not job_folder.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return job_folder

    return get_latest_job_folder()


# -----------------------------
# File saving
# -----------------------------

async def save_upload(file: Optional[UploadFile], input_folder: Path):
    if file is None:
        return None

    file_path = input_folder / file.filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


# -----------------------------
# Text extraction functions
# -----------------------------

def extract_text_from_pdf(pdf_path):
    extracted_pages = []

    pdf_doc = fitz.open(pdf_path)

    for page_number, page in enumerate(pdf_doc, start=1):
        page_text = page.get_text()

        if page_text.strip():
            extracted_pages.append(f"[PDF Page {page_number}]\n{page_text}")

    pdf_doc.close()

    return "\n\n".join(extracted_pages)


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    image_text = pytesseract.image_to_string(image)

    return f"[Image Text]\n{image_text}"


def extract_text_from_audio(audio_path):
    model = get_whisper()
    result = model.transcribe(str(audio_path))

    transcript = result.get("text", "")

    return f"[Audio Transcript]\n{transcript}"


def extract_text_from_video(video_path, job_folder):
    audio_path = job_folder / "extracted_video_audio.wav"

    video_clip = VideoFileClip(str(video_path))

    if video_clip.audio is None:
        video_clip.close()
        raise ValueError("Video has no audio track.")

    video_clip.audio.write_audiofile(str(audio_path), logger=None)
    video_clip.close()

    model = get_whisper()
    result = model.transcribe(str(audio_path))

    transcript = result.get("text", "")

    return f"[Video Transcript]\n{transcript}"


# -----------------------------
# Cleaning / chunking / embeddings
# -----------------------------

def clean_text(raw_text):
    text = raw_text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"[-_=]{4,}", " ", text)

    clean_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            clean_lines.append("")
            continue

        if len(line) <= 2:
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        raise ValueError("No words found after cleaning.")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "start_word": start,
            "end_word": min(end, len(words)),
            "text": " ".join(chunk_words)
        })

        chunk_id += 1
        start += chunk_size - chunk_overlap

    return chunks


def generate_embeddings(chunks):
    model = get_embedder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# -----------------------------
# Save/load processed data
# -----------------------------

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_files(job_folder, raw_text, cleaned_text, chunks, embeddings, index, metadata):
    with open(job_folder / "all_extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(job_folder / "clean_text.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    save_json(job_folder / "chunks.json", chunks)

    np.save(job_folder / "embeddings.npy", embeddings)

    faiss.write_index(index, str(job_folder / "faiss.index"))

    save_json(job_folder / "processing_metadata.json", metadata)


def load_rag_assets(job_folder):
    chunks_path = job_folder / "chunks.json"
    index_path = job_folder / "faiss.index"

    if not chunks_path.exists() or not index_path.exists():
        raise HTTPException(
            status_code=400,
            detail="This job has not been processed yet. Missing chunks.json or faiss.index."
        )

    chunks = load_json(chunks_path)
    index = faiss.read_index(str(index_path))

    return chunks, index


# -----------------------------
# RAG functions
# -----------------------------

def embed_query(question):
    model = get_embedder()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(query_embedding).astype("float32")


def retrieve_top_chunks(question, chunks, index, top_k=3):
    query_embedding = embed_query(question)

    scores, indices = index.search(query_embedding, top_k)

    retrieved = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]

        retrieved.append({
            "chunk_id": chunk["chunk_id"],
            "score": float(score),
            "text": chunk["text"]
        })

    return retrieved


def format_context(retrieved_chunks):
    parts = []

    for chunk in retrieved_chunks:
        parts.append(
            f"[Chunk {chunk['chunk_id']} | Score {chunk['score']:.4f}]\n{chunk['text']}"
        )

    return "\n\n".join(parts)


def run_llm(prompt, max_new_tokens=250):
    tokenizer, model = get_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_with_rag(question, job_folder, top_k=3):
    chunks, index = load_rag_assets(job_folder)

    retrieved_chunks = retrieve_top_chunks(
        question=question,
        chunks=chunks,
        index=index,
        top_k=top_k
    )

    context = format_context(retrieved_chunks)

    prompt = f"""
You are a helpful academic assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"I could not find this information in the uploaded material."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = run_llm(prompt, max_new_tokens=200)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }


# -----------------------------
# Feature prompts
# -----------------------------

def generate_feature(job_folder, feature_name, prompt_question, top_k=6, max_new_tokens=300):
    chunks, index = load_rag_assets(job_folder)

    retrieved_chunks = retrieve_top_chunks(
        question=prompt_question,
        chunks=chunks,
        index=index,
        top_k=top_k
    )

    context = format_context(retrieved_chunks)

    prompt = f"""
You are an academic learning assistant.

Use ONLY the context below.
Do not use outside knowledge.
Be structured, clear, and useful.

Context:
{context}

Task:
{prompt_question}

Output:
"""

    output = run_llm(prompt, max_new_tokens=max_new_tokens)

    feature_folder = job_folder / "features"
    feature_folder.mkdir(exist_ok=True)

    save_path = feature_folder / f"{feature_name}.json"

    result = {
        "feature": feature_name,
        "output": output,
        "retrieved_chunks": retrieved_chunks
    }

    save_json(save_path, result)

    return result


# -----------------------------
# API endpoints
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "AI Study Assistant Backend is running",
        "endpoints": [
            "POST /ingest",
            "POST /ask",
            "GET /summary",
            "GET /flashcards",
            "GET /quiz",
            "GET /knowledge-graph"
        ]
    }


@app.post("/ingest")
async def ingest_files(
    pdf: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    job_id, job_folder, input_folder, _ = create_job_folder()

    extracted_parts = []
    input_files = {}

    pdf_path = await save_upload(pdf, input_folder)
    image_path = await save_upload(image, input_folder)
    audio_path = await save_upload(audio, input_folder)
    video_path = await save_upload(video, input_folder)

    try:
        if pdf_path:
            extracted_parts.append(extract_text_from_pdf(pdf_path))
            input_files["pdf"] = str(pdf_path)

        if image_path:
            extracted_parts.append(extract_text_from_image(image_path))
            input_files["image"] = str(image_path)

        if audio_path:
            extracted_parts.append(extract_text_from_audio(audio_path))
            input_files["audio"] = str(audio_path)

        if video_path:
            extracted_parts.append(extract_text_from_video(video_path, job_folder))
            input_files["video"] = str(video_path)

        if not extracted_parts:
            raise HTTPException(status_code=400, detail="Upload at least one file.")

        raw_text = "\n\n".join(extracted_parts)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        embeddings = generate_embeddings(chunks)
        index = build_faiss_index(embeddings)

        metadata = {
            "job_id": job_id,
            "input_files": input_files,
            "num_chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_shape": list(embeddings.shape),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "vector_index": "FAISS IndexFlatIP",
            "llm_model": LLM_MODEL_NAME
        }

        save_processed_files(
            job_folder=job_folder,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            chunks=chunks,
            embeddings=embeddings,
            index=index,
            metadata=metadata
        )

        return {
            "message": "Files ingested and processed successfully",
            "job_id": job_id,
            "job_folder": str(job_folder),
            "num_chunks": len(chunks),
            "embedding_shape": list(embeddings.shape)
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/ask")
def ask_question(
    question: str = Form(...),
    job_id: Optional[str] = Form(None),
    top_k: int = Form(3)
):
    job_folder = get_job_folder(job_id)

    result = answer_with_rag(
        question=question,
        job_folder=job_folder,
        top_k=top_k
    )

    return {
        "job_id": job_folder.name,
        "question": question,
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"]
    }


@app.get("/summary")
def detailed_summary(job_id: Optional[str] = None):
    job_folder = get_job_folder(job_id)

    prompt_question = """
Create a detailed structured summary of this content.

Include:
1. Main topic
2. Key concepts
3. Important definitions
4. Step-by-step explanation
5. Practical examples
6. Things to remember
7. Possible exam topics
"""

    result = generate_feature(
        job_folder=job_folder,
        feature_name="summary",
        prompt_question=prompt_question,
        top_k=6,
        max_new_tokens=350
    )

    return {
        "job_id": job_folder.name,
        "summary": result["output"],
        "retrieved_chunks": result["retrieved_chunks"]
    }


@app.get("/flashcards")
def flashcards(
    job_id: Optional[str] = None,
    num_cards: int = 5
):
    job_folder = get_job_folder(job_id)

    prompt_question = f"""
Create exactly {num_cards} flashcards from this content.

Format:
Flashcard 1
Front: ...
Back: ...

Rules:
- Use only the uploaded content
- Keep the front side short
- Keep the back side clear
- Focus on important concepts and definitions
"""

    result = generate_feature(
        job_folder=job_folder,
        feature_name="flashcards",
        prompt_question=prompt_question,
        top_k=6,
        max_new_tokens=350
    )

    return {
        "job_id": job_folder.name,
        "flashcards": result["output"],
        "retrieved_chunks": result["retrieved_chunks"]
    }


@app.get("/quiz")
def quiz(
    job_id: Optional[str] = None,
    difficulty: str = "easy",
    num_questions: int = 5
):
    difficulty = difficulty.lower()

    if difficulty not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail="Difficulty must be easy, medium, or hard."
        )

    job_folder = get_job_folder(job_id)

    prompt_question = f"""
Create exactly {num_questions} quiz questions from this content.

Difficulty: {difficulty}

Difficulty rules:
- Easy: direct recall and definition-based questions
- Medium: conceptual understanding and small application questions
- Hard: analytical, inference-based, and deeper reasoning questions

Format:
Q1. ...
A. ...
B. ...
C. ...
D. ...
Correct Answer: ...
Explanation: ...

Use only the uploaded content.
"""

    result = generate_feature(
        job_folder=job_folder,
        feature_name=f"quiz_{difficulty}",
        prompt_question=prompt_question,
        top_k=6,
        max_new_tokens=400
    )

    return {
        "job_id": job_folder.name,
        "difficulty": difficulty,
        "quiz": result["output"],
        "retrieved_chunks": result["retrieved_chunks"]
    }


@app.get("/knowledge-graph")
def knowledge_graph(job_id: Optional[str] = None):
    job_folder = get_job_folder(job_id)

    prompt_question = """
Extract a knowledge graph from this content.

Return the result in this format:

Entities:
- Entity name: short description

Relationships:
- Entity A -> relationship -> Entity B

Rules:
- Use only the uploaded content
- Focus on important concepts
- Keep relationships meaningful
- Avoid duplicate entities
"""

    result = generate_feature(
        job_folder=job_folder,
        feature_name="knowledge_graph",
        prompt_question=prompt_question,
        top_k=8,
        max_new_tokens=400
    )

    return {
        "job_id": job_folder.name,
        "knowledge_graph": result["output"],
        "retrieved_chunks": result["retrieved_chunks"]
    }


@app.get("/jobs/latest")
def latest_job():
    job_folder = get_latest_job_folder()

    metadata_path = job_folder / "processing_metadata.json"

    metadata = {}

    if metadata_path.exists():
        metadata = load_json(metadata_path)

    return {
        "latest_job_id": job_folder.name,
        "job_folder": str(job_folder),
        "metadata": metadata
    }