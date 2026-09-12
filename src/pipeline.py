"""End-to-end customer support agent pipeline orchestrator."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import sys

from src.intent import IntentClassifier, IntentResult
from src.retrieval import KnowledgeRetriever, RetrievedDocument
from src.reply import ReplyGenerator, ReplyResult
from src.escalate import EscalationManager, EscalationDecision


@dataclass
class PipelineResponse:
    """Final output of the support agent pipeline."""
    query: str
    intent: str
    intent_confidence: float
    is_escalated: bool
    escalation_reason: Optional[str]
    escalation_queue: Optional[str]
    reply: str
    cited_docs: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupportAgentPipeline:
    """Orchestrates intent classification, retrieval, escalation, and reply generation."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        reply_generator: Optional[ReplyGenerator] = None,
        escalation_manager: Optional[EscalationManager] = None,
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or KnowledgeRetriever()
        self.reply_generator = reply_generator or ReplyGenerator()
        self.escalation_manager = escalation_manager or EscalationManager()

    def process(self, query: str) -> PipelineResponse:
        """Process an incoming customer query through the agent pipeline.

        Args:
            query: Customer input text.

        Returns:
            PipelineResponse containing intent, escalation state, and reply.
        """
        # Step 1: Classify Intent
        intent_res: IntentResult = self.classifier.classify(query)

        # Step 2: Retrieve Relevant Knowledge Base Articles
        retrieved_docs: list[RetrievedDocument] = self.retriever.retrieve(query, top_k=3)

        # Step 3: Evaluate Escalation Conditions
        escalation: EscalationDecision = self.escalation_manager.evaluate(
            query, intent_res, retrieved_docs
        )

        # Step 4: Handle Escalation or Generate Reply
        if escalation.should_escalate:
            reply_text = (
                f"I am routing your request to our {escalation.suggested_queue} team. "
                f"A support specialist will assist you shortly. (Reference: {escalation.reason.value})"
            )
            return PipelineResponse(
                query=query,
                intent=intent_res.primary_intent.value,
                intent_confidence=intent_res.confidence,
                is_escalated=True,
                escalation_reason=escalation.reason.value,
                escalation_queue=escalation.suggested_queue,
                reply=reply_text,
                cited_docs=[],
            )

        # Generate Grounded AI Reply
        reply_res: ReplyResult = self.reply_generator.generate(query, intent_res, retrieved_docs)

        return PipelineResponse(
            query=query,
            intent=intent_res.primary_intent.value,
            intent_confidence=intent_res.confidence,
            is_escalated=False,
            escalation_reason=None,
            escalation_queue=None,
            reply=reply_res.text,
            cited_docs=reply_res.grounded_in_docs,
        )


if __name__ == "__main__":
    sample_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I want to return an item I received 5 days ago. What is your refund policy?"
    )
    pipeline = SupportAgentPipeline()
    response = pipeline.process(sample_query)
    print("\n--- Pipeline Execution Output ---")
    print(json.dumps(response.to_dict(), indent=2))
