# VectorDB Python — Build a Vector Database from Scratch in Python

A fully working **Vector Database** built from scratch in Python with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python/hnswlib)        ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **3 things** installed:

1. **Python 3.8+** (with pip)
2. **Ollama** (runs the local AI models)
3. **Git** (optional, for cloning)

---

## Step-by-Step Setup

### Step 1 — Install Python

**macOS / Linux:**
```bash
# Check if Python 3.8+ is installed
python3 --version

# If not installed, use your package manager
# macOS:
brew install python3

# Ubuntu/Debian:
sudo apt-get install python3 python3-pip python3-venv
```

**Windows:**
1. Go to **https://www.python.org/downloads** and download Python 3.10+
2. During installation, **check "Add Python to PATH"**
3. After install, open **PowerShell** and verify:
```powershell
python --version
pip --version
```

---

### Step 2 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and click **Download**
2. Run the installer for your OS
3. Ollama starts automatically in the background
4. Open a **terminal/PowerShell** and pull the two required models:

```bash
ollama pull nomic-embed-text
```
*(~274 MB — this is the embedding model)*

```bash
ollama pull llama3.2
```
*(~2 GB — this is the language model)*

5. Verify Ollama is running:
```bash
ollama list
```
You should see both models listed.

> **Minimum specs for Ollama:** 8GB RAM recommended. The models will use ~3GB total.

---

### Step 3 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB-Python.git
cd VectorDB-Python
```

Or just copy the files manually if you don't have git.

---

### Step 4 — Create a Python Virtual Environment (Recommended)

This keeps dependencies isolated from your system Python.

```bash
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate.bat
```

Your terminal prompt should now show `(venv)` at the start.

---

### Step 5 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Flask** — REST API server
- **Flask-CORS** — Cross-origin requests
- **numpy** — Numerical computing
- **scikit-learn** — KD-Tree implementation
- **hnswlib** — HNSW algorithm
- **requests** — HTTP client for Ollama

Installation takes ~2-3 minutes depending on your internet.

---

### Step 6 — Run Everything

**Terminal 1** — Start Ollama (if not already running):
```bash
ollama serve
```
*(If Ollama is already in your system background, skip this)*

**Terminal 2** — Start the VectorDB server:
```bash
# Make sure you're in the venv first (see Step 4)
python main.py
```

You should see:
```
=== VectorDB Engine ===
http://localhost:5000
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

**Open your browser** and go to:
```
http://localhost:5000
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based only on your documents
```

The answer streams in with a typewriter effect. Click the **context chips** to see exactly which chunks the AI used.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:5000`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```bash
curl "http://localhost:5000/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```bash
curl -X POST http://localhost:5000/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
VectorDB-Python/
├── main.py         ← Flask backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html      ← Frontend (PCA scatter plot, chat UI, benchmark)
├── requirements.txt ← Python dependencies
└── README_PYTHON.md ← This file
```

### Architecture (main.py)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Implemented via the **hnswlib** library (used by production systems). Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Implemented via **scikit-learn**'s KDTree. Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions).

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `pip: command not found` | Make sure Python 3.8+ is installed and in your PATH |
| Port 5000 already in use | Change the port in `main.py` line `app.run(port=5000)` |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on a laptop CPU. Use llama3.2:1b for faster answers |

### Use a Smaller/Faster LLM

If llama3.2 is too slow on your laptop, switch to the 1B model:

```bash
ollama pull llama3.2:1b
```

Then edit `main.py` and change:
```python
self.gen_model = "llama3.2:1b"   # line ~287 in OllamaClient.__init__
```

Restart the server.

---

## Deactivate Virtual Environment

When you're done, deactivate the virtual environment:

```bash
# macOS / Linux / Windows PowerShell:
deactivate
```

Next time you run the app, just activate the venv again (Step 4).

---

## License

MIT — use this however you want.

---

## Differences from C++ Version

- **REST API:** Flask (Python) instead of cpp-httplib (C++)
- **Distance metrics:** NumPy for faster computation
- **KD-Tree:** scikit-learn instead of hand-written
- **HNSW:** hnswlib (C++ library via Python bindings) — identical performance to C++
- **Port:** 5000 (Flask default) instead of 8080
- **Deployment:** Single file `main.py` instead of compiled binary

All functionality, algorithms, and API endpoints are identical.
