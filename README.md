# 📄 Local Document Diff Agent (Offline LLM System)

A fully offline, privacy-first document comparison engine powered by local LLM inference.  
It detects structural and semantic differences between two documents and presents a structured compliance report through a Streamlit interface.

---

## 🧠 Overview

This system compares two documents (PDF, DOCX, TXT) and performs:

- 📌 Structural diffing (added / removed clauses)  
- 🧠 Semantic change detection using a local LLM  
- ⚖️ Risk classification of clause modifications  
- 📊 Interactive UI for reviewing differences  

All processing happens locally on your machine—no cloud APIs, no data transmission.

---

## 🏗️ Architecture

```text
Document A ─┐
            ├── Parser (PDF/DOCX/Text)
Document B ─┘
         ↓
Clause Segmentation Engine
         ↓
Structural Diff Engine
         ↓
Local LLM (Qwen / Phi via Ollama)
         ↓
Structured JSON Output
         ↓
Streamlit UI Dashboard
```

---

## 🔒 Privacy Model

✔ Fully offline execution
✔ No external APIs or cloud inference
✔ No telemetry or logging services
✔ All data remains on local machine memory
⚙️ Tech Stack
Python 3.10+
Streamlit
PyMuPDF
docx2txt
Ollama
Local LLMs (e.g., Qwen, Phi)

---

## 🤖 Supported Models

Run locally via Ollama:

ollama pull qwen2.5:7b

Other supported models:

Phi-4 Mini
Mistral Small
Llama variants

---

## 📦 Installation

1. Clone repository

git clone https://github.com/your-username/local-doc-diff-agent.git

cd local-doc-diff-agent

2. Install dependencies

pip install -r requirements.txt

3. Install Ollama

Download from:
https://ollama.com

Then pull model:

ollama pull qwen2.5:7b

---

## 🚀 Running the Application

streamlit run app.py

Then open:

http://localhost:8501

---

## 📂 Project Structure

local-doc-diff-agent/

│

├── app.py              # Streamlit UI

├── agent.py            # Core orchestration engine

├── parser.py          # Document parsing + clause extraction

├── llm.py             # Local LLM inference layer

├── schemas.py         # Structured output schema

├── requirements.txt

└── README.md

---

## 🧪 Features

✔ Structural Diffing

Detects:

Added clauses

Removed clauses

✔ Semantic Change Detection

Uses local LLM to classify:

Wording changes

Obligation shifts

No material change

✔ Risk Scoring

Each modified clause is labeled:

Low

Medium

High

✔ Fully Offline Execution

No internet required after model download.

🧰 Example Output

{
  "change_type": "Obligation Shifted",

  "summary": "Payment timeline changed from 30 days to 60 days",

  "risk": "High"
}

---

## 🔐 Use Cases

Legal contract review

Compliance audits

Procurement comparison

Policy version tracking

Internal document governance

---

## ⚠️ Disclaimer

This tool is intended for assistance purposes only and does not replace professional legal review.