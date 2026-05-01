import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def chunk_document(doc: dict) -> list:
    enc = _get_encoder()
    tokens = enc.encode(doc["text"])
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(tokens):
        window = tokens[i : i + CHUNK_SIZE]
        chunk_text = enc.decode(window)
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_c{chunk_index}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": chunk_text,
                "chunk_index": chunk_index,
            }
        )
        i += step
        chunk_index += 1
    return chunks


def chunk_documents(docs: list) -> list:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
