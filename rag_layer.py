import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


OUTPUT_DIR = Path("data/outputs")


def get_latest_uuid_folder(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    folders = [folder for folder in output_dir.iterdir() if folder.is_dir()]

    if not folders:
        raise FileNotFoundError("No UUID folders found in data/outputs")

    latest_folder = max(folders, key=lambda folder: folder.stat().st_mtime)

    return latest_folder


def load_chunks(job_folder):
    chunks_path = job_folder / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def load_faiss_index(job_folder):
    index_path = job_folder / "faiss.index"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    index = faiss.read_index(str(index_path))

    return index


def load_metadata(job_folder):
    metadata_path = job_folder / "processing_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def load_embedding_model(model_name):
    embedder = SentenceTransformer(model_name)
    return embedder


def embed_query(question, embedder):
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.array(query_embedding).astype("float32")

    return query_embedding


def retrieve_top_chunks(question, chunks, index, embedder, top_k=3):
    query_embedding = embed_query(question, embedder)

    scores, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]

        retrieved_chunks.append({
            "chunk_id": chunk.get("chunk_id", int(chunk_index)),
            "score": float(score),
            "text": chunk["text"]
        })

    return retrieved_chunks


def format_context(retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Chunk {chunk['chunk_id']} | Score {chunk['score']:.4f}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return context


def load_llm(model_name="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def generate_answer(question, retrieved_chunks, tokenizer, model):
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


class RAGQueryEngine:
    def __init__(
        self,
        output_dir=OUTPUT_DIR,
        llm_model_name="google/flan-t5-small"
    ):
        self.job_folder = get_latest_uuid_folder(output_dir)

        self.metadata = load_metadata(self.job_folder)

        self.embedding_model_name = self.metadata.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chunks = load_chunks(self.job_folder)
        self.index = load_faiss_index(self.job_folder)

        self.embedder = load_embedding_model(self.embedding_model_name)

        self.tokenizer, self.model = load_llm(llm_model_name)

    def retrieve(self, question, top_k=3):
        retrieved_chunks = retrieve_top_chunks(
            question=question,
            chunks=self.chunks,
            index=self.index,
            embedder=self.embedder,
            top_k=top_k
        )

        return retrieved_chunks

    def ask(self, question, top_k=3):
        retrieved_chunks = self.retrieve(
            question=question,
            top_k=top_k
        )

        answer = generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            tokenizer=self.tokenizer,
            model=self.model
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "source_folder": str(self.job_folder)
        }


def print_result(result):
    print("\nQUESTION")
    print(result["question"])

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCE FOLDER")
    print(result["source_folder"])

    print("\nRETRIEVED CHUNKS")
    for chunk in result["retrieved_chunks"]:
        print("\n" + "-" * 60)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Score: {chunk['score']:.4f}")
        print(chunk["text"][:500])


if __name__ == "__main__":
    rag = RAGQueryEngine(
        llm_model_name="google/flan-t5-small"
    )

    question = "What is this content about?"

    result = rag.ask(
        question=question,
        top_k=3
    )

    print_result(result)