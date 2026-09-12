# Hiver AI Customer Support Agent

An intelligent, production-oriented AI customer support agent architecture built with Python 3.11. The system combines intent classification, knowledge retrieval (RAG), automated grounded response generation, and safety-first escalation routing.

---

## Directory Structure

```text
hiver-support-agent/
├── data/                  # Raw and sampled customer support CSVs (gitignored)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── intent.py          # Customer intent taxonomy & classification
│   ├── retrieval.py       # Knowledge base search & article grounding
│   ├── reply.py           # Empathetic, grounded response generation
│   ├── escalate.py        # Multi-tiered human handoff & escalation policy
│   └── pipeline.py        # End-to-end support pipeline orchestrator
├── eval/
│   ├── __init__.py
│   ├── golden_set.csv     # Labeled benchmark test dataset
│   ├── judge.py           # Evaluation runner & pipeline evaluator
│   └── metrics.py         # Precision, Recall, F1, Accuracy, Grounding
├── notebooks/             # Data exploration & error analysis (gitignored data)
│   └── .gitkeep
├── report.md              # Technical assignment report and evaluation metrics
├── README.md              # Project documentation and quickstart guide
├── requirements.txt       # Project dependencies
├── .env.example           # Example configuration and API keys
└── .gitignore             # Excludes data/*.csv, .env, __pycache__, .venv, etc.
```

---

## Prerequisites

- **Python 3.11+**
- `pip` / `virtualenv`
- (Optional) OpenAI API Key for live LLM completions

---

## Quickstart Setup

### 1. Clone or navigate to the project
```bash
cd hiver-support-agent
```

### 2. Create and activate a Python 3.11 Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Update `.env` with your API keys if you wish to run live LLM completions. If no API key is provided, the agent automatically falls back to safe local deterministic synthesis.

---

## Usage

### Run Pipeline via CLI
Test individual customer queries against the support pipeline:

```bash
python -m src.pipeline "Where is my package? I ordered it 3 days ago."
```

Example Output:
```json
{
  "query": "Where is my package? I ordered it 3 days ago.",
  "intent": "order_status",
  "intent_confidence": 0.87,
  "is_escalated": false,
  "escalation_reason": null,
  "escalation_queue": null,
  "reply": "Thank you for contacting support! Regarding your inquiry: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days...",
  "cited_docs": ["KB-101"]
}
```

### Test Escalation Behavior
```bash
python -m src.pipeline "I demand to speak to a human manager immediately!"
```

Output:
```json
{
  "query": "I demand to speak to a human manager immediately!",
  "intent": "escalation_request",
  "intent_confidence": 0.95,
  "is_escalated": true,
  "escalation_reason": "explicit_human_request",
  "escalation_queue": "General Support",
  "reply": "I am routing your request to our General Support team. A support specialist will assist you shortly. (Reference: explicit_human_request)",
  "cited_docs": []
}
```

---

## Benchmark Evaluation

Run the evaluation suite against `eval/golden_set.csv`:

```bash
python -m eval.judge
```

This runs all test cases, measures accuracy, escalation precision/recall/F1, grounding overlap, and outputs a summary report saved to `eval/eval_results.json`.

---

## Data & Notebooks

- Place raw datasets (e.g. `twcs.csv`, `sample.csv`) into the `data/` directory.
- `data/*.csv` is ignored by git to keep repository size lean and prevent leaking private data.
- Use `notebooks/` for exploratory data analysis (EDA), clustering, and prompt experimentation.
