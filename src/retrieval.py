"""Knowledge base retrieval and contextual search module."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import re


@dataclass
class RetrievedDocument:
    """A retrieved knowledge article snippet."""
    doc_id: str
    title: str
    content: str
    relevance_score: float
    category: str


# Default baseline knowledge base for customer support
DEFAULT_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "doc_id": "KB-101",
        "title": "Order Tracking & Delivery Times",
        "category": "order_status",
        "content": (
            "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. "
            "You can track your package by entering your tracking number on our portal at support.example.com/track. "
            "Tracking updates can take up to 24 hours to reflect after shipment."
        ),
    },
    {
        "doc_id": "KB-102",
        "title": "Return & Refund Policy",
        "category": "refund_return",
        "content": (
            "Customers may request returns within 30 days of delivery. Items must be unused and in original packaging. "
            "Refunds are processed to the original payment method within 5-7 business days after our warehouse inspects the returned item."
        ),
    },
    {
        "doc_id": "KB-103",
        "title": "Password Reset & Account Recovery",
        "category": "account_access",
        "content": (
            "To reset your password, visit the login page and click 'Forgot Password'. A secure reset link will be sent "
            "to your registered email address. If you have two-factor authentication enabled and lost your device, "
            "our security team must verify your identity manually."
        ),
    },
    {
        "doc_id": "KB-104",
        "title": "Subscription Cancellation & Billing Cycles",
        "category": "cancellation",
        "content": (
            "Subscriptions renew automatically on the billing date. You can cancel anytime from Settings > Billing > Manage Subscription. "
            "Upon cancellation, your access remains active until the end of the current billing cycle. No prorated refunds are issued for partial months."
        ),
    },
    {
        "doc_id": "KB-105",
        "title": "Technical Troubleshooting & Browser Compatibility",
        "category": "technical_issue",
        "content": (
            "If you experience errors or blank screens, please clear your browser cache and cookies or try Incognito mode. "
            "We support the latest versions of Chrome, Safari, Firefox, and Edge. If issues persist, check status.example.com for outages."
        ),
    },
    {
        "doc_id": "KB-106",
        "title": "Escalation to Human Support Specialist",
        "category": "escalation_request",
        "content": (
            "For complex billing disputes, enterprise accounts, or unresolved technical issues, customer tickets can be escalated "
            "to a Tier-2 Human Support Specialist. Human agents are available Monday through Friday from 9 AM to 6 PM EST."
        ),
    },
]


class KnowledgeRetriever:
    """Retrieves relevant support documents using keyword and semantic search heuristics."""

    def __init__(self, documents: Optional[List[Dict[str, str]]] = None):
        self.documents = documents or DEFAULT_KNOWLEDGE_BASE

    def _tokenize(self, text: str) -> set[str]:
        """Simple word tokenization for lexical search."""
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve the top-k most relevant knowledge articles for a customer query.

        Args:
            query: Customer question or message.
            top_k: Number of documents to return.

        Returns:
            List of RetrievedDocument objects ranked by relevance score.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_docs: List[RetrievedDocument] = []

        for doc in self.documents:
            doc_text = f"{doc['title']} {doc['content']} {doc.get('category', '')}".lower()
            doc_tokens = self._tokenize(doc_text)

            # Jaccard / Overlap lexical similarity score
            intersection = query_tokens.intersection(doc_tokens)
            score = len(intersection) / (math.sqrt(len(query_tokens) * len(doc_tokens)) + 1e-5)

            # Boost exact title matches
            title_tokens = self._tokenize(doc["title"].lower())
            if query_tokens.intersection(title_tokens):
                score += 0.25

            if score > 0.05:
                scored_docs.append(
                    RetrievedDocument(
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        content=doc["content"],
                        relevance_score=round(min(score, 1.0), 3),
                        category=doc.get("category", "general"),
                    )
                )

        scored_docs.sort(key=lambda d: d.relevance_score, reverse=True)
        return scored_docs[:top_k]
