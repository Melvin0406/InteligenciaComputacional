import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Vector store (mismo código que el notebook)
class Document:
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

class SearchResult:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document

class FilteredVectorStore:
    def __init__(self, model: SentenceTransformer):
        self._model = model
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def add_documents(self, documents: list[Document]):
        texts = [d.text for d in documents]
        emb = self._model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-10)
        self._documents.extend(documents)
        self._embeddings = emb if self._embeddings is None else np.vstack([self._embeddings, emb])

    def search(self, query: str, top_k: int = 5,
               metadata_filter: dict | None = None) -> list[SearchResult]:
        if not self._documents:
            return []
        indices = [
            i for i, d in enumerate(self._documents)
            if not metadata_filter or all(d.metadata.get(k) == v for k, v in metadata_filter.items())
        ]
        if not indices:
            return []
        q = self._model.encode([query], convert_to_numpy=True)
        q = q / np.maximum(np.linalg.norm(q, keepdims=True), 1e-10)
        scores = (self._embeddings[indices] @ q.T).flatten()
        top = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(float(scores[j]), self._documents[indices[j]]) for j in top]

# Estado global (en memoria)
class AppState:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.fvs = FilteredVectorStore(self.model)
        # Almacena el documento original completo: id -> {"text": str, "metadata": dict}
        self.originals: dict[str, dict] = {}
        # Mapea cada chunk a su documento original: chunk_doc_id -> original_id
        self.chunk_to_original: dict[str, str] = {}

state = AppState()

# Esquema de metadatos
class ArticleMetadata(BaseModel):
    title: str
    author: str
    category: str
    source: str

# App
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # el modelo ya se cargó en AppState


app = FastAPI(title="Article Vector Store API", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok", "documents_indexed": len(state.originals)}


# Modelos de request / response
class ArticleCreate(BaseModel):
    text: str
    metadata: ArticleMetadata


class ArticleCreateResponse(BaseModel):
    id: str
    chunks_created: int


# Chunking
CHUNK_SIZE = 400
SPLIT_THRESHOLD = 500

def split_text(text: str) -> list[str]:
    if len(text) <= SPLIT_THRESHOLD:
        return [text]
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


# POST /documents
@app.post("/documents", response_model=ArticleCreateResponse, status_code=201)
def create_document(body: ArticleCreate):
    doc_id = str(uuid.uuid4())

    # Guardar el documento original completo
    state.originals[doc_id] = {
        "text": body.text,
        "metadata": body.metadata.model_dump(),
    }

    # Generar chunks e indexarlos
    chunks = split_text(body.text)
    chunk_docs = []
    for chunk_text in chunks:
        chunk_id = str(uuid.uuid4())
        chunk_metadata = {**body.metadata.model_dump(), "original_id": doc_id}
        chunk_docs.append(Document(text=chunk_text, metadata=chunk_metadata))
        state.chunk_to_original[chunk_id] = doc_id

    state.fvs.add_documents(chunk_docs)

    return ArticleCreateResponse(id=doc_id, chunks_created=len(chunks))
