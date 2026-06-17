"""rag_retriever.py - 전조 증상 검색 모듈 (단계 3).

로컬 ChromaDB + BAAI/bge-m3 임베딩으로, camera_id 와 event_time 기준 과거 시간창에서
화재 전조(연기·스파크·과열·배회 등) 텍스트 이력을 시맨틱 검색한다.
cctv_memory 의 TextEmbedder / VectorStore 패턴을 이식하되, ChromaDB EmbeddingFunction
인터페이스의 버전별 변동을 피하려고 임베딩을 직접 계산해 embeddings/query_embeddings 로 넘긴다.

CLI 스모크: python -m src.rag_retriever <camera_id> <event_time_iso> ["질의"]
"""
import threading
from datetime import datetime

from . import config

_STORE = None
_STORE_LOCK = threading.Lock()


# ── 시간 유틸 ────────────────────────────────────────────────────────────────
def _to_epoch(t):
    """event_time(ISO str / epoch 숫자 / datetime) -> epoch 초(float). (naive=로컬 기준 일관 사용)"""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    if isinstance(t, str):
        s = t.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            return float(s)  # "1718600000" 같은 epoch 문자열
    raise TypeError(f"지원하지 않는 event_time 타입: {type(t)}")


def _fmt_time(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M:%S")


# ── 임베딩 (bge-m3) ──────────────────────────────────────────────────────────
class TextEmbedder:
    """BAAI/bge-m3 임베더. VLM 과 VRAM 충돌 회피 위해 기본 CPU. normalize=True(코사인용)."""

    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or config.EMBED_MODEL_ID
        self.device = device or config.EMBED_DEVICE
        self._model = None

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EMBED] 로딩: {self.model_id} device={self.device}", flush=True)
            try:
                # safetensors 우선: torch<2.6 에서 .bin(torch.load)이 CVE 로 차단되는 문제 회피.
                self._model = SentenceTransformer(
                    self.model_id, device=self.device,
                    model_kwargs={"use_safetensors": True})
            except Exception:
                self._model = SentenceTransformer(self.model_id, device=self.device)
        return self

    def encode(self, texts):
        """str -> list[float], list[str] -> list[list[float]] (정규화 벡터)."""
        if self._model is None:
            self.load()
        single = isinstance(texts, str)
        arr = self._model.encode([texts] if single else list(texts),
                                 normalize_embeddings=True)
        return arr[0].tolist() if single else arr.tolist()


# ── 이력 저장소 (ChromaDB) ────────────────────────────────────────────────────
class HistoryStore:
    """과거 CCTV 묘사 이력 컬렉션. 메타: {camera_id, event_time(epoch), source}."""

    def __init__(self, embedder=None):
        import chromadb
        self.embedder = embedder or TextEmbedder()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})

    def count(self):
        return self.col.count()

    def add_logs(self, records):
        """records: [{camera_id, event_time, caption, [id], [source]}] -> 추가 개수."""
        if not records:
            return 0
        ids, docs, metas = [], [], []
        for i, r in enumerate(records):
            ev = _to_epoch(r["event_time"])
            cam = r["camera_id"]
            ids.append(r.get("id") or f"{cam}:{int(ev)}:{i}")
            docs.append(r["caption"])
            metas.append({"camera_id": cam, "event_time": ev, "source": r.get("source", "")})
        embs = self.embedder.encode(docs)
        self.col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        return len(ids)

    def search(self, query, k=5, where=None):
        """query + where -> [{id, document, metadata, score}] (score=코사인 유사도=1-distance)."""
        n = self.col.count()
        if n == 0:
            return []
        qemb = self.embedder.encode(query)
        res = self.col.query(query_embeddings=[qemb],
                             n_results=min(k, n), where=where or None)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": round(1.0 - res["distances"][0][i], 4),
            })
        return out

    def reset(self):
        """컬렉션 비우고 재생성(시드 재주입용)."""
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.col = self.client.get_or_create_collection(
            config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def get_store():
    """프로세스 공유 단일 HistoryStore(임베더 1회 로드)."""
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = HistoryStore()
    return _STORE


# ── 공개 API ─────────────────────────────────────────────────────────────────
def retrieve_precursors(camera_id, event_time, query=None, k=None):
    """camera_id + event_time 기준 과거 시간창에서 화재 전조 이력을 시맨틱 검색.

    시간창: [event_time - LOOKBACK_MAX_HOURS, event_time - LOOKBACK_MIN_HOURS].
    Returns: [{caption, camera_id, event_time(epoch), time(str), score}] (score 내림차순, min_score 필터).
    """
    store = get_store()
    ev = _to_epoch(event_time)
    t_hi = ev - config.LOOKBACK_MIN_HOURS * 3600.0
    t_lo = ev - config.LOOKBACK_MAX_HOURS * 3600.0
    where = {"$and": [
        {"camera_id": {"$eq": camera_id}},
        {"event_time": {"$gte": t_lo}},
        {"event_time": {"$lte": t_hi}},
    ]}
    hits = store.search(query or config.PRECURSOR_QUERY, k=k or config.RAG_TOP_K, where=where)
    out = []
    for h in hits:
        if h["score"] < config.SEARCH_MIN_SCORE:
            continue
        md = h["metadata"]
        out.append({
            "caption": h["document"],
            "camera_id": md.get("camera_id"),
            "event_time": md.get("event_time"),
            "time": _fmt_time(md.get("event_time")),
            "score": h["score"],
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 3:
        print('usage: python -m src.rag_retriever <camera_id> <event_time_iso> ["질의"]')
        raise SystemExit(1)
    cam, ev = sys.argv[1], sys.argv[2]
    q = sys.argv[3] if len(sys.argv) > 3 else None
    res = retrieve_precursors(cam, ev, query=q)
    print(f"[store count] {get_store().count()}")
    print(json.dumps(res, ensure_ascii=False, indent=2))
