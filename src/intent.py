"""Intent classification module for incoming customer support messages."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import os


class CustomerIntent(str, Enum):
    """Standardized taxonomy of customer support intents."""
    ORDER_STATUS = "order_status"
    REFUND_RETURN = "refund_return"
    ACCOUNT_ACCESS = "account_access"
    BILLING_SUBSCRIPTION = "billing_subscription"
    TECHNICAL_ISSUE = "technical_issue"
    CANCELLATION = "cancellation"
    PRODUCT_INQUIRY = "product_inquiry"
    ESCALATION_REQUEST = "escalation_request"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: CustomerIntent
    confidence: float
    secondary_intents: List[CustomerIntent]
    rationale: str
    urgency_level: str  # e.g., low, medium, high, critical


class IntentClassifier:
    """Classifies user queries into discrete support intents with confidence scores."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def classify(self, text: str) -> IntentResult:
        """Classify the incoming customer query text.

        Args:
            text: Raw customer message text.

        Returns:
            IntentResult with primary intent, confidence, and urgency.
        """
        # Rule-based fast path / keyword heuristics for fallback
        lower_text = text.lower()

        if any(w in lower_text for w in ["human", "agent", "representative", "manager", "lawyer", "sue"]):
            return IntentResult(
                primary_intent=CustomerIntent.ESCALATION_REQUEST,
                confidence=0.95,
                secondary_intents=[CustomerIntent.GENERAL_QUERY],
                rationale="Direct request for human assistance or legal mention detected.",
                urgency_level="critical" if "lawyer" in lower_text or "sue" in lower_text else "high"
            )

        if any(w in lower_text for w in ["refund", "return", "money back", "reimburse"]):
            return IntentResult(
                primary_intent=CustomerIntent.REFUND_RETURN,
                confidence=0.88,
                secondary_intents=[CustomerIntent.BILLING_SUBSCRIPTION],
                rationale="Refund/return keywords present in customer message.",
                urgency_level="medium"
            )

        if any(w in lower_text for w in ["cancel", "unsubscribe", "stop subscription"]):
            return IntentResult(
                primary_intent=CustomerIntent.CANCELLATION,
                confidence=0.85,
                secondary_intents=[CustomerIntent.BILLING_SUBSCRIPTION],
                rationale="Cancellation intent keywords detected.",
                urgency_level="high"
            )

        if any(w in lower_text for w in ["password", "login", "locked out", "sign in", "2fa"]):
            return IntentResult(
                primary_intent=CustomerIntent.ACCOUNT_ACCESS,
                confidence=0.90,
                secondary_intents=[CustomerIntent.TECHNICAL_ISSUE],
                rationale="Authentication and account access keywords detected.",
                urgency_level="medium"
            )

        if any(w in lower_text for w in ["order", "track", "delivery", "shipping", "shipped", "package"]):
            return IntentResult(
                primary_intent=CustomerIntent.ORDER_STATUS,
                confidence=0.87,
                secondary_intents=[CustomerIntent.GENERAL_QUERY],
                rationale="Order fulfillment or shipping keywords detected.",
                urgency_level="medium"
            )

        if any(w in lower_text for w in ["error", "bug", "crash", "down", "not working", "broken"]):
            return IntentResult(
                primary_intent=CustomerIntent.TECHNICAL_ISSUE,
                confidence=0.84,
                secondary_intents=[CustomerIntent.GENERAL_QUERY],
                rationale="Technical malfunction or error keywords detected.",
                urgency_level="high"
            )

        return IntentResult(
            primary_intent=CustomerIntent.GENERAL_QUERY,
            confidence=0.60,
            secondary_intents=[],
            rationale="Generic inquiry; no specific category trigger matched.",
            urgency_level="low"
        )
