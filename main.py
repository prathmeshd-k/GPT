#!/usr/bin/env python3
"""
VectorDB — Vector Database in Python
Implements HNSW, KD-Tree, and Brute Force search algorithms with REST API
"""

import json
import math
import time
import random
import requests
import heapq
from typing import List, Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from sklearn.neighbors import KDTree as SKLKDTree
import hnswlib

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    """Single vector with metadata"""
    id: int
    metadata: str
    category: str
    emb: List[float]


@dataclass
class DocItem:
    """Document chunk with embedding"""
    id: int
    title: str
    text: str
    emb: List[float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity (converted to distance: 1 - similarity)"""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: List[float], b: List[float]) -> float:
    """Manhattan distance"""
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> Callable:
    """Get distance function by name"""
    if metric == "cosine":
        return cosine
    elif metric == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    """Simple brute force O(N) search"""
    
    def __init__(self):
        self.items: Dict[int, VectorItem] = {}
    
    def insert(self, item: VectorItem):
        """Insert item"""
        self.items[item.id] = item
    
    def remove(self, item_id: int):
        """Remove item by ID"""
        if item_id in self.items:
            del self.items[item_id]
    
    def knn(self, q: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """Find k nearest neighbors"""
        distances = []
        for item_id, item in self.items.items():
            d = dist_fn(q, item.emb)
            distances.append((d, item_id))
        distances.sort()
        return distances[:k]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDTree:
    """KD-Tree for spatial partitioning"""
    
    def __init__(self, dims: int):
        self.dims = dims
        self.items: Dict[int, VectorItem] = {}
        self.tree: Optional[SKLKDTree] = None
        self.tree_ids: List[int] = []
    
    def insert(self, item: VectorItem):
        """Insert item and rebuild tree"""
        self.items[item.id] = item
        self._rebuild()
    
    def remove(self, item_id: int):
        """Remove item and rebuild tree"""
        if item_id in self.items:
            del self.items[item_id]
            self._rebuild()
    
    def _rebuild(self):
        """Rebuild the KD-tree from current items"""
        if not self.items:
            self.tree = None
            self.tree_ids = []
            return
        
        self.tree_ids = sorted(self.items.keys())
        data = np.array([self.items[item_id].emb for item_id in self.tree_ids], dtype=np.float32)
        self.tree = SKLKDTree(data, leaf_size=40)
    
    def knn(self, q: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """Find k nearest neighbors using KD-tree"""
        if self.tree is None or not self.items:
            return []
        
        q_arr = np.array([q], dtype=np.float32)
        distances, indices = self.tree.query(q_arr, k=min(k, len(self.tree_ids)))
        
        result = []
        for dist, idx in zip(distances[0], indices[0]):
            item_id = self.tree_ids[idx]
            result.append((float(dist), item_id))
        
        return result


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    """HNSW implementation using hnswlib for production performance"""
    
    def __init__(self, dims: int, M: int = 16, ef_construction: int = 200):
        self.dims = dims
        self.M = M
        self.ef_construction = ef_construction
        self.items: Dict[int, VectorItem] = {}
        self.index: Optional[hnswlib.Index] = None
        self.id_to_index: Dict[int, int] = {}  # item_id -> hnsw index position
        self.index_to_id: Dict[int, int] = {}  # hnsw index position -> item_id
        self.next_index = 0
    
    def insert(self, item: VectorItem, dist_fn: Callable):
        """Insert item into HNSW"""
        self.items[item.id] = item
        
        # Reinitialize index if needed
        if self.index is None or self.next_index == 0:
            # Choose space based on metric (L2 for euclidean, IP for cosine, etc.)
            self.index = hnswlib.Index(space='cosine', dim=self.dims)
            self.index.init_index(max_elements=len(self.items) + 10000, ef_construction=self.ef_construction, M=self.M)
        
        idx = self.next_index
        self.id_to_index[item.id] = idx
        self.index_to_id[idx] = item.id
        
        emb_array = np.array(item.emb, dtype=np.float32)
        self.index.add_items(emb_array.reshape(1, -1), np.array([idx]))
        self.next_index += 1
    
    def remove(self, item_id: int):
        """Remove item from HNSW"""
        if item_id in self.items:
            del self.items[item_id]
            if item_id in self.id_to_index:
                del self.id_to_index[item_id]
            # Note: hnswlib doesn't support efficient removal, so we just mark as deleted
    
    def knn(self, q: List[float], k: int, ef: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        """Find k nearest neighbors using HNSW"""
        if self.index is None or self.next_index == 0:
            return []
        
        q_array = np.array([q], dtype=np.float32)
        self.index.set_ef(max(ef, k))
        
        try:
            labels, distances = self.index.knn_query(q_array, k=min(k, self.next_index))
            result = []
            for dist, hnsw_idx in zip(distances[0], labels[0]):
                if hnsw_idx in self.index_to_id:
                    item_id = self.index_to_id[hnsw_idx]
                    if item_id in self.items:
                        result.append((float(dist), item_id))
            return result
        except Exception:
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get graph info for visualization"""
        return {
            "topLayer": 0,
            "nodeCount": len(self.items),
            "nodesPerLayer": [len(self.items)],
            "edgesPerLayer": [0],
            "nodes": [
                {"id": item.id, "metadata": item.metadata, "category": item.category, "maxLyr": 0}
                for item in self.items.values()
            ],
            "edges": []
        }


# =====================================================================
#  VECTOR DATABASE (demo 16D index)
# =====================================================================

class VectorDB:
    """Main vector database combining all three algorithms"""
    
    def __init__(self, dims: int = DIMS):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(dims)
        self.lock = threading.Lock()
        self.next_id = 1
    
    def insert(self, metadata: str, category: str, emb: List[float], dist_fn: Callable) -> int:
        """Insert vector into all three indexes"""
        with self.lock:
            item = VectorItem(self.next_id, metadata, category, emb)
            self.next_id += 1
            
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            
            return item.id
    
    def remove(self, item_id: int) -> bool:
        """Remove vector from all three indexes"""
        with self.lock:
            if item_id not in self.store:
                return False
            
            del self.store[item_id]
            self.bf.remove(item_id)
            self.kdt.remove(item_id)
            self.hnsw.remove(item_id)
            return True
    
    def search(self, q: List[float], k: int, metric: str, algo: str) -> Dict[str, Any]:
        """Search using specified algorithm and metric"""
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()
            
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:  # hnsw
                raw = self.hnsw.knn(q, k, 50, dist_fn)
            
            elapsed_us = int((time.perf_counter() - t0) * 1e6)
            
            hits = []
            for dist, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": float(dist),
                        "embedding": item.emb
                    })
            
            return {
                "results": hits,
                "latencyUs": elapsed_us,
                "algo": algo,
                "metric": metric
            }
    
    def benchmark(self, q: List[float], k: int, metric: str) -> Dict[str, Any]:
        """Benchmark all three algorithms"""
        with self.lock:
            dist_fn = get_dist_fn(metric)
            
            def time_algo(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1e6)
            
            bf_us = time_algo(lambda: self.bf.knn(q, k, dist_fn))
            kd_us = time_algo(lambda: self.kdt.knn(q, k, dist_fn))
            hnsw_us = time_algo(lambda: self.hnsw.knn(q, k, 50, dist_fn))
            
            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }
    
    def all(self) -> List[VectorItem]:
        """Get all items"""
        with self.lock:
            return list(self.store.values())
    
    def get_hnsw_info(self) -> Dict[str, Any]:
        """Get HNSW graph info"""
        with self.lock:
            return self.hnsw.get_info()
    
    def size(self) -> int:
        """Get number of items"""
        with self.lock:
            return len(self.store)


# =====================================================================
#  OLLAMA CLIENT — wraps local Ollama REST API
# =====================================================================

class OllamaClient:
    """HTTP client for Ollama embedding and generation API"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"
        self.base_url = f"http://{host}:{port}"
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def embed(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            payload = {"model": self.embed_model, "prompt": text}
            resp = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=30)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("embedding", [])
        except Exception:
            return []
    
    def generate(self, prompt: str) -> str:
        """Generate text from Ollama"""
        try:
            payload = {"model": self.gen_model, "prompt": prompt, "stream": False}
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            if resp.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            data = resp.json()
            return data.get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    """Database for real document embeddings using HNSW"""
    
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw: Optional[HNSW] = None
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0
    
    def insert(self, title: str, text: str, emb: List[float]) -> int:
        """Insert document chunk"""
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
                self.hnsw = HNSW(self.dims)
            
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            
            self.store[item.id] = item
            
            # Create VectorItem for indexing
            vec_item = VectorItem(item.id, title, "doc", emb)
            self.bf.insert(vec_item)
            if self.hnsw:
                self.hnsw.insert(vec_item, cosine)
            
            return item.id
    
    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        """Search for similar documents"""
        with self.lock:
            if not self.store:
                return []
            
            # Use brute force for small datasets, HNSW for large
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            elif self.hnsw:
                raw = self.hnsw.knn(q, k, 50, cosine)
            else:
                raw = self.bf.knn(q, k, cosine)
            
            result = []
            for dist, item_id in raw:
                if item_id in self.store and dist <= max_dist:
                    result.append((float(dist), self.store[item_id]))
            
            return result
    
    def remove(self, item_id: int) -> bool:
        """Remove document"""
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            if self.hnsw:
                self.hnsw.remove(item_id)
            return True
    
    def all(self) -> List[DocItem]:
        """Get all documents"""
        with self.lock:
            return list(self.store.values())
    
    def size(self) -> int:
        """Get number of documents"""
        with self.lock:
            return len(self.store)
    
    def get_dims(self) -> int:
        """Get embedding dimension"""
        return self.dims


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    
    chunks = []
    step = chunk_words - overlap_words
    
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    
    return chunks


# =====================================================================
#  DEMO DATA
# =====================================================================

DEMO_DATA = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
]


def load_demo(db: VectorDB):
    """Load demo vectors into database"""
    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_DATA:
        db.insert(metadata, category, emb, dist_fn)


# =====================================================================
#  FLASK APP & REST API
# =====================================================================

app = Flask(__name__)
CORS(app)

# Initialize databases
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

# Load demo data
load_demo(db)

# Print startup info
print("=== VectorDB Engine ===")
print("http://localhost:5000")
print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
ollama_up = ollama.is_available()
print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
if ollama_up:
    print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.route('/search', methods=['GET'])
def search():
    """Search for vectors: /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw"""
    try:
        v_str = request.args.get('v', '')
        q = [float(x) for x in v_str.split(',') if x.strip()]
        
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400
        
        k = int(request.args.get('k', 5))
        metric = request.args.get('metric', 'cosine')
        algo = request.args.get('algo', 'hnsw')
        
        result = db.search(q, k, metric, algo)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/insert', methods=['POST'])
def insert():
    """Insert a vector"""
    try:
        data = request.get_json()
        metadata = data.get('metadata', '')
        category = data.get('category', '')
        emb = data.get('embedding', [])
        
        if not metadata or not emb or len(emb) != DIMS:
            return jsonify({"error": "invalid body"}), 400
        
        item_id = db.insert(metadata, category, emb, get_dist_fn("cosine"))
        return jsonify({"id": item_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/delete/<int:item_id>', methods=['DELETE'])
def delete(item_id):
    """Delete a vector by ID"""
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route('/items', methods=['GET'])
def items():
    """Get all items"""
    all_items = db.all()
    return jsonify([asdict(item) for item in all_items])


@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Benchmark all three algorithms"""
    try:
        v_str = request.args.get('v', '')
        q = [float(x) for x in v_str.split(',') if x.strip()]
        
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400
        
        k = int(request.args.get('k', 5))
        metric = request.args.get('metric', 'cosine')
        
        result = db.benchmark(q, k, metric)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/hnsw-info', methods=['GET'])
def hnsw_info():
    """Get HNSW graph structure info"""
    info = db.get_hnsw_info()
    return jsonify(info)


@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    })


# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

@app.route('/doc/insert', methods=['POST'])
def doc_insert():
    """Embed and insert document"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        
        if not title or not text:
            return jsonify({"error": "need title and text"}), 400
        
        chunks = chunk_text(text, 250, 30)
        ids = []
        
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return jsonify({
                    "error": "Ollama unavailable. Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2"
                }), 503
            
            chunk_title = (f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title)
            chunk_id = doc_db.insert(chunk_title, chunk, emb)
            ids.append(chunk_id)
        
        return jsonify({
            "ids": ids,
            "chunks": len(chunks),
            "dims": doc_db.get_dims()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/doc/delete/<int:doc_id>', methods=['DELETE'])
def doc_delete(doc_id):
    """Delete a document"""
    ok = doc_db.remove(doc_id)
    return jsonify({"ok": ok})


@app.route('/doc/list', methods=['GET'])
def doc_list():
    """List all documents"""
    all_docs = doc_db.all()
    result = []
    for doc in all_docs:
        preview = doc.text[:120]
        if len(doc.text) > 120:
            preview += "…"
        word_count = len(doc.text.split())
        result.append({
            "id": doc.id,
            "title": doc.title,
            "preview": preview,
            "words": word_count
        })
    return jsonify(result)


@app.route('/doc/search', methods=['POST'])
def doc_search():
    """Search documents"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        k = data.get('k', 3)
        
        if not question:
            return jsonify({"error": "need question"}), 400
        
        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503
        
        hits = doc_db.search(q_emb, k)
        
        contexts = []
        for dist, doc_item in hits:
            contexts.append({
                "id": doc_item.id,
                "title": doc_item.title,
                "distance": float(dist)
            })
        
        return jsonify({"contexts": contexts})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/doc/ask', methods=['POST'])
def doc_ask():
    """RAG pipeline: embed question -> retrieve context -> generate answer"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        k = data.get('k', 3)
        
        if not question:
            return jsonify({"error": "need question"}), 400
        
        # Step 1: embed question
        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503
        
        # Step 2: retrieve top-k chunks
        hits = doc_db.search(q_emb, k)
        
        # Step 3: build prompt
        context_parts = []
        for i, (dist, doc_item) in enumerate(hits):
            context_parts.append(f"[{i+1}] {doc_item.title}:\n{doc_item.text}\n")
        
        context = "\n".join(context_parts)
        
        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
            "'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # Step 4: generate answer
        answer = ollama.generate(prompt)
        
        # Step 5: return everything
        contexts = []
        for dist, doc_item in hits:
            contexts.append({
                "id": doc_item.id,
                "title": doc_item.title,
                "text": doc_item.text,
                "distance": float(dist)
            })
        
        return jsonify({
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": contexts,
            "docCount": doc_db.size()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size()
    })


# Serve index.html
@app.route('/', methods=['GET'])
def index():
    """Serve frontend"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)