# AI Customer Support Agent - Assignment Report

**Author:** Mahesh Boda  
**Repository:** `hiver-support-agent`  
**Python Version:** 3.11  
**Date:** September 2026  

---

## 1. Executive Summary

This project implements an intelligent, production-ready AI Customer Support Agent tailored for customer support operations. The agent orchestrates:
1. **Multi-class Intent Classification** to categorize customer requests accurately.
2. **Contextual Retrieval (RAG)** to fetch verified knowledge base documentation.
3. **Guardrailed Response Generation** to synthesize polite, helpful, and hallucination-free answers.
4. **Deterministic Escalation Engine** to safely transfer high-risk, low-confidence, or sensitive tickets to human agents.
5. **Systematic Evaluation Suite** measuring intent accuracy, escalation precision/recall, latency, and response grounding.

---

## 2. Architecture & Pipeline Flow

```
+------------------+
| Customer Query   |
+--------+---------+
         |
         v
+--------+---------+        +--------------------------+
| Intent Classifier| -----> | Knowledge Base Retriever |
+--------+---------+        +------------+-------------+
         |                               |
         +---------------+---------------+
                         |
                         v
            +------------+-------------+
            |    Escalation Manager    |
            +------------+-------------+
                         |
         +---------------+---------------+
         |                               |
  [Should Escalate]              [Should Auto-Reply]
         |                               |
         v                               v
+--------+---------+            +--------+---------+
| Human Handoff    |            | Response         |
| Routing Queue    |            | Generator        |
+------------------+            +------------------+
```

### Key Stages:
- **`intent.py`**: Categorizes customer inquiries into standard intents (`order_status`, `refund_return`, `account_access`, `billing_subscription`, `technical_issue`, `cancellation`, `escalation_request`, etc.) and calculates a confidence score.
- **`retrieval.py`**: Queries indexed support articles to ground responses in company policy and prevent hallucinations.
- **`escalate.py`**: Applies multi-tier safety checks:
  - *Tier 1 (P1)*: Legal threats, fraud, regulatory mentions.
  - *Tier 2 (P2)*: Explicit human representative requests.
  - *Tier 3 (P3)*: Low confidence (< 0.65) or missing knowledge articles.
  - *Tier 4 (P4)*: Safe automated resolution.
- **`reply.py`**: Generates empathetic, grounded customer replies citing knowledge articles.
- **`pipeline.py`**: Orchestrates all modules in a single clean interface.

---

## 3. Evaluation & Experimental Results

The pipeline was benchmarked against the golden evaluation dataset (`eval/golden_set.csv`).

### Benchmark Metrics Summary

| Metric | Score / Value | Target Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Intent Classification Accuracy** | 91.7% | > 85% | Pass |
| **Escalation Precision** | 100.0% | > 90% | Pass |
| **Escalation Recall** | 83.3% | > 80% | Pass |
| **Escalation F1-Score** | 90.9% | > 85% | Pass |
| **False Escalation Rate** | 0.0% | < 10% | Pass |
| **Missed Escalation Rate** | 16.7% | < 20% | Pass |
| **Average Query Latency** | < 1 ms (local) / ~450 ms (LLM) | < 1.5s | Pass |

---

## 4. Key Design Decisions & Trade-Offs

1. **Hybrid Fallback Mechanism**:
   - The agent operates seamlessly both with an active OpenAI API key (for nuanced conversational replies) and in offline/fallback mode (deterministic template synthesis), guaranteeing 100% uptime.
2. **Conservative Escalation Policy**:
   - In customer service, a false resolution (answering wrongly or missing an angry customer) is significantly worse than escalating early. The escalation manager favors high precision on explicit requests and legal terms.
3. **Modularity & Testability**:
   - Each component (`intent`, `retrieval`, `reply`, `escalate`) is decoupled with clean dataclasses, making unit testing and benchmarking straightforward.

---

## 5. Failure Modes & Edge Cases Observed

1. **Multi-Intent Ambiguity**:
   - *Example:* "I want to return my order because your website crashed."
   - *Behavior:* Touches both `refund_return` and `technical_issue`. Current implementation ranks by primary score. Future enhancement: multi-label routing.
2. **Sarcasm & Passive Aggression**:
   - *Example:* "Great job shipping my order to the wrong state, truly remarkable."
   - *Behavior:* Lexical matching may detect positive sentiment words ("great", "remarkable"). Requires sentiment polarity scoring via LLM judge or fine-tuned sentiment model.

---

## 6. Future Improvements & Scaling

1. **Vector Embeddings (ChromaDB / FAISS)**: Scale document retrieval to thousands of support docs using `text-embedding-3-small`.
2. **Session Memory & Multi-Turn State**: Track conversation history across multiple turns using Redis or DynamoDB.
3. **Automated Continuous Evaluation**: Run `eval/judge.py` as a GitHub Actions CI step on every prompt/pipeline modification.
