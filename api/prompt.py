from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI
from pinecone import Pinecone


# ================= CONFIG =================
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
TOP_K = 10

# These must exist in Vercel env vars
LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL")  # must end with /v1
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


# ================= PROMPTS =================
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only "
    "based on the TED dataset context provided to you (metadata and transcript passages). "
    "You must not use any external knowledge, the open internet, or information that is "
    "not explicitly contained in the retrieved context. If the answer cannot be determined "
    "from the provided context, respond: \"I don’t know based on the provided TED data.\" "
    "Always explain your answer using the given context, quoting or paraphrasing the "
    "relevant transcript or metadata when helpful."
)


# ================= HANDLER =================
class handler(BaseHTTPRequestHandler):

    # ---------- init helpers (kept close to rag_test.py) ----------
    def _init_openai_client(self) -> OpenAI:
        if not LLMOD_API_KEY or not LLMOD_BASE_URL:
            raise RuntimeError("Missing LLMOD_API_KEY or LLMOD_BASE_URL environment variables.")
        return OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)

    def _init_pinecone_index(self):
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME environment variables.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)

    # ---------- HTTP ----------
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            question = (data.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "Missing 'question' field"})
                return

            payload = self.run_rag(question)
            self._send_json(200, payload)

        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ================= RAG =================
    def run_rag(self, question: str):
        client = self._init_openai_client()
        index = self._init_pinecone_index()

        # 1) Embed question (same spirit as rag_test.py)
        q_vec = self.embed_query(client, question)

        # 2) Retrieve
        matches = self.retrieve(index, q_vec, top_k=TOP_K)

        # 3) Build augmented user prompt with context
        user_prompt = self.build_user_prompt(question, matches)

        # 4) Generate answer (chat)
        answer = self.chat_completion(client, SYSTEM_PROMPT, user_prompt)

        # 5) Required response schema
        return {
            "response": answer,
            "context": [
                {
                    "talk_id": (m.get("metadata") or {}).get("talk_id"),
                    "title": (m.get("metadata") or {}).get("title"),
                    "chunk": (m.get("metadata") or {}).get("text", ""),
                    "score": m.get("score"),
                }
                for m in matches
            ],
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt,
            },
        }

    # ---------- Embeddings (rag_test.py-like) ----------
    def embed_query(self, client: OpenAI, text: str):
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    # ---------- Retrieval (Pinecone SDK, rag_test.py-like) ----------
    def retrieve(self, index, vector, top_k: int):
        res = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        return res.get("matches", [])

    # ---------- Chat completion (compatible with rag_test.py approach) ----------
    def chat_completion(self, client: OpenAI, system_prompt: str, user_prompt: str) -> str:
        # IMPORTANT: do NOT set temperature=0.0 (your provider rejected it)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    # ---------- Prompt building ----------
    def build_user_prompt(self, question: str, matches):
        blocks = []
        for i, m in enumerate(matches, start=1):
            meta = m.get("metadata") or {}
            blocks.append(
                f"[{i}] talk_id={meta.get('talk_id')}, "
                f"title={meta.get('title')}, "
                f"speaker={meta.get('speaker_1')}, "
                f"chunk_index={meta.get('chunk_index')}\n"
                f"{meta.get('text', '')}"
            )

        context_text = "\n\n".join(blocks) if blocks else "(no context retrieved)"

        return f"""
            You are given several context chunks from TED talks. Use only this context to answer the question.
            
            Question:
            {question}
            
            Context:
            {context_text}
            
            Now provide a concise answer based only on this context.
            If the answer cannot be determined from the context, respond exactly with:
            "I don’t know based on the provided TED data."
            """.strip()

    # ---------- Response helper ----------
    def _send_json(self, status: int, payload: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))
