# VectorDB Python — Quick Start (5 Minutes)

## Prerequisites

- **Python 3.8+** installed (check: `python --version`)
- **Ollama** installed and running (check: `ollama list` shows models)

Already have Ollama models? Skip to Step 2.

---

## Step 1: Set Up Ollama (2 min)

If you don't have Ollama running:

```bash
# Install from https://ollama.com
# Then in any terminal:
ollama pull nomic-embed-text
ollama pull llama3.2
```

Keep this terminal open. Ollama runs in background.

---

## Step 2: Create Virtual Environment (30 sec)

```bash
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 3: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

Wait for completion.

---

## Step 4: Run the Server (10 sec)

```bash
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

---

## Step 5: Open in Browser (10 sec)

Go to: **http://localhost:5000**

Done! 🎉

---

## What You Can Do

### Tab 1: Search Demo Vectors
- Type: `binary tree`, `sushi`, `basketball`
- Choose algorithm: HNSW, KD-Tree, Brute Force
- See results + comparison

### Tab 2: Upload Documents
- Paste any text (lecture notes, articles, etc.)
- Click "EMBED & INSERT"
- Documents get split into chunks and embedded with Ollama

### Tab 3: Ask AI
- Type a question about your uploaded documents
- AI finds relevant chunks and generates an answer

---

## API Examples

```bash
# Search via curl
curl "http://localhost:5000/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3"

# Insert a vector
curl -X POST http://localhost:5000/insert \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": "My vector",
    "category": "test",
    "embedding": [0.9, 0.8, ..., 0.1]  // 16 floats
  }'

# Ask AI (RAG)
curl -X POST http://localhost:5000/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is dynamic programming?", "k": 3}'
```

---

## Troubleshooting

**"Ollama: OFFLINE"**
→ Run `ollama serve` in a new terminal

**"ModuleNotFoundError: flask"**
→ Run `pip install -r requirements.txt`

**Port 5000 already in use**
→ Edit `main.py`, change `port=5000` to `port=5001`

**LLM answer is slow**
→ Normal! Takes 10-30s on laptop. Use `ollama pull llama3.2:1b` for faster 1B model.

---

## Stop the Server

```bash
Ctrl+C
```

To deactivate the virtual environment:
```bash
deactivate
```

---

## See Also

- **README_PYTHON.md** — Full documentation
- **CONVERSION_GUIDE.md** — C++ to Python details
- **main.py** — Full source code (well-commented)

---

## Architecture

```
Browser (index.html)
    ↓
Flask Server (main.py:5000)
    ├── HNSW (hnswlib) → fast vector search
    ├── KD-Tree (scikit-learn) → spatial index
    ├── Brute Force → exact search
    └── Ollama (REST) → embeddings + LLM
```

All three algorithms run side-by-side. Compare their speed!

---

Happy vectoring! 🚀
