"""Escalation detection and human handoff routing module."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import os
from src.intent import IntentResult, CustomerIntent
from src.retrieval import RetrievedDocument


class EscalationReason(str, Enum):
    """Reasons why an interaction is escalated to human agents."""
    EXPLICIT_HUMAN_REQUEST = "explicit_human_request"
    LOW_CONFIDENCE = "low_confidence_score"
    HIGH_NEGATIVE_SENTIMENT = "high_negative_sentiment"
    LEGAL_OR_COMPLIANCE = "legal_or_compliance_threat"
    MISSING_KNOWLEDGE = "no_relevant_documentation"
    CRITICAL_URGENCY = "critical_urgency_level"
    NONE = "none"


@dataclass
class EscalationDecision:
    """Outcome of escalation evaluation."""
    should_escalate: bool
    reason: EscalationReason
    priority: str  # P1, P2, P3, P4
    suggested_queue: str  # e.g., Billing, TechSupport, Legal, General
    explanation: str


class EscalationManager:
    """Determines whether a support conversation requires human intervention."""

    def __init__(self, confidence_threshold: float = 0.65):
        env_thresh = os.getenv("ESCALATION_CONFIDENCE_THRESHOLD")
        self.confidence_threshold = float(env_thresh) if env_thresh else confidence_threshold

    def evaluate(
        self,
        customer_query: str,
        intent_result: IntentResult,
        retrieved_docs: List[RetrievedDocument],
    ) -> EscalationDecision:
        """Evaluate whether to escalate the conversation to a human agent.

        Args:
            customer_query: Original customer query.
            intent_result: Classified intent and confidence.
            retrieved_docs: List of retrieved knowledge documents.

        Returns:
            EscalationDecision object with routing metadata.
        """
        lower_query = customer_query.lower()

        # 1. Legal / Compliance trigger (Highest Priority - P1)
        if any(term in lower_query for term in ["lawyer", "attorney", "sue", "legal action", "court", "fraud"]):
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LEGAL_OR_COMPLIANCE,
                priority="P1",
                suggested_queue="Legal & Risk",
                explanation="Customer mentioned legal action or fraud.",
            )

        # 2. Explicit human agent request
        if intent_result.primary_intent == CustomerIntent.ESCALATION_REQUEST or any(
            req in lower_query for req in ["human agent", "speak to someone", "representative", "real person"]
        ):
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.EXPLICIT_HUMAN_REQUEST,
                priority="P2",
                suggested_queue="General Support",
                explanation="Customer explicitly requested a human representative.",
            )

        # 3. Critical urgency flagged by intent
        if intent_result.urgency_level == "critical":
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.CRITICAL_URGENCY,
                priority="P1",
                suggested_queue="Emergency Response",
                explanation="Ticket marked with critical urgency level.",
            )

        # 4. Low intent classification confidence
        if intent_result.confidence < self.confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                priority="P3",
                suggested_queue="Triage",
                explanation=f"Intent confidence ({intent_result.confidence:.2f}) is below threshold ({self.confidence_threshold:.2f}).",
            )

        # 5. Missing knowledge / zero retrieved documents with acceptable score
        if not retrieved_docs or retrieved_docs[0].relevance_score < 0.15:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.MISSING_KNOWLEDGE,
                priority="P3",
                suggested_queue="Tier-2 Specialist",
                explanation="No relevant knowledge base articles found to resolve query.",
            )

        # Default: No escalation needed; AI can handle
        return EscalationDecision(
            should_escalate=False,
            reason=EscalationReason.NONE,
            priority="P4",
            suggested_queue="AI Automated",
            explanation="Query can be safely resolved by AI assistant.",
        )
