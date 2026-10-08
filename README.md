# VectorDB Python — AI Vector Database & RAG Chatbot

A fully working **Vector Database built with Python** that demonstrates how vector search engines work internally.

The project implements **HNSW, KD-Tree, and Brute Force** search algorithms and integrates **Ollama** for real document embeddings and local LLM-based question answering through a **RAG (Retrieval-Augmented Generation) pipeline**.

---

## 🚀 Features

* 🔍 HNSW vector search
* 🌳 KD-Tree search
* ⚡ Brute Force exact search
* 📐 Cosine, Euclidean and Manhattan distance metrics
* 🧠 Real document embeddings using Ollama
* 🤖 RAG-based AI chatbot
* 📄 Automatic document chunking
* 📊 PCA-based vector visualization
* 🌐 Flask REST API
* 💬 Interactive web interface
* 🐳 Docker and Docker Compose support
* 🔒 Thread-safe data operations

The application includes 20 pre-loaded 16-dimensional demo vectors across Computer Science, Mathematics, Food and Sports categories.

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │   Web Browser     │
                         │    index.html     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Flask REST API  │
                         │     main.py      │
                         └────────┬─────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
        ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
        │    HNSW     │   │  KD-Tree    │   │ Brute Force │
        │  hnswlib    │   │ scikit-learn│   │   Python    │
        └─────────────┘   └─────────────┘   └─────────────┘
                │
                ▼
        ┌──────────────────┐
        │      Ollama      │
        │                  │
        │ nomic-embed-text │
        │     llama3.2     │
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │       RAG        │
        │ Retrieve Context │
        │   Generate Answer│
        └──────────────────┘
```

---

## 🧠 How the RAG Pipeline Works

```text
User Question
      │
      ▼
nomic-embed-text
      │
      ▼
Question Embedding
      │
      ▼
HNSW Vector Search
      │
      ▼
Relevant Document Chunks
      │
      ▼
Context + Question
      │
      ▼
llama3.2
      │
      ▼
AI Generated Answer
```

The project converts text into 768-dimensional embeddings using `nomic-embed-text`, searches for semantically similar document chunks, and sends the retrieved context to `llama3.2` to generate the answer.

---

## 🔎 Search Algorithms

### 1. HNSW

**Hierarchical Navigable Small World**

HNSW is used for fast approximate nearest-neighbor vector search.

The Python version uses the `hnswlib` library.

### 2. KD-Tree

KD-Tree is used for spatial partitioning and nearest-neighbor searches.

The implementation uses `scikit-learn`'s KDTree.

### 3. Brute Force

Brute Force calculates the distance between the query vector and every stored vector.

```text
Query
  │
  ├── Vector 1 → Distance
  ├── Vector 2 → Distance
  ├── Vector 3 → Distance
  ├── ...
  └── Vector N → Distance

Sort distances
      ↓
Return Top-K
```

This provides exact search but becomes expensive as the number of vectors increases.

---

## 📐 Distance Metrics

The project supports:

| Metric    | Description                                 |
| --------- | ------------------------------------------- |
| Cosine    | Measures angular similarity between vectors |
| Euclidean | Measures straight-line distance             |
| Manhattan | Measures absolute coordinate differences    |

Cosine distance is also used for the real document embedding workflow.

---

## 📄 Document Processing

Users can paste documents into the application.

The system:

```text
Document
   ↓
Text Chunking
   ↓
Overlapping Chunks
   ↓
Ollama Embedding
   ↓
768D Vectors
   ↓
HNSW Index
   ↓
Semantic Search
```

Long documents are divided into overlapping chunks before embedding. This allows the RAG system to retrieve smaller, relevant portions of a document rather than processing the entire document at once.

---

## 🌐 REST API

### Vector APIs

| Method | Endpoint      | Description         |
| ------ | ------------- | ------------------- |
| GET    | `/search`     | K-NN vector search  |
| POST   | `/insert`     | Insert a vector     |
| DELETE | `/delete/:id` | Delete a vector     |
| GET    | `/items`      | List vectors        |
| GET    | `/benchmark`  | Compare algorithms  |
| GET    | `/hnsw-info`  | HNSW information    |
| GET    | `/stats`      | Database statistics |

### Document & RAG APIs

| Method | Endpoint          | Description              |
| ------ | ----------------- | ------------------------ |
| POST   | `/doc/insert`     | Embed and store document |
| GET    | `/doc/list`       | List documents           |
| DELETE | `/doc/delete/:id` | Delete document          |
| POST   | `/doc/search`     | Search documents         |
| POST   | `/doc/ask`        | Ask AI using RAG         |
| GET    | `/status`         | System status            |

The uploaded project documentation lists these endpoints as part of the Python implementation.

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Flask-CORS
* NumPy
* scikit-learn
* hnswlib
* Requests

These dependencies are defined in the project's requirements files.

### AI

* Ollama
* `nomic-embed-text`
* `llama3.2`

### Frontend

* HTML
* CSS
* JavaScript
* PCA visualization
* Interactive vector search UI

The frontend provides search, document embedding and AI/RAG functionality.

### Deployment

* Docker
* Docker Compose

Docker Compose runs the Flask VectorDB application together with Ollama.

---

## 📁 Project Structure

```text
VectorDB-Python/
│
├── main.py
├── index.html
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── README.md
├── QUICKSTART.md
├── CONVERSION_GUIDE.md
└── SUMMARY.md
```

The Python conversion documentation identifies `main.py` as the main Flask server and `index.html` as the frontend.

---

# ⚙️ Installation

## Prerequisites

Install:

1. Python 3.8+
2. Ollama
3. Git — optional

The project documentation recommends Python 3.8+ and Ollama.

---

## 1. Install Ollama Models

Install Ollama and download the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify:

```bash
ollama list
```

The application uses `nomic-embed
