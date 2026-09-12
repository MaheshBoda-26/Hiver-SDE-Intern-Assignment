"""Response generation module for customer support replies."""

from dataclasses import dataclass
from typing import List, Optional
import os
from src.retrieval import RetrievedDocument
from src.intent import IntentResult


@dataclass
class ReplyResult:
    """Structured response payload generated for the customer."""
    text: str
    grounded_in_docs: List[str]  # List of doc_ids used
    tone: str  # empathetic, informative, apologetic
    model_used: str


class ReplyGenerator:
    """Generates polite, accurate, and context-grounded replies."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate(
        self,
        customer_query: str,
        intent_result: IntentResult,
        retrieved_docs: List[RetrievedDocument],
    ) -> ReplyResult:
        """Generate a grounded response for the customer.

        Args:
            customer_query: Original customer query.
            intent_result: Classified intent and urgency.
            retrieved_docs: Relevant retrieved knowledge base snippets.

        Returns:
            ReplyResult with final reply text and citation metadata.
        """
        doc_ids = [doc.doc_id for doc in retrieved_docs]

        # Check if OpenAI client can be invoked
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)

                context_str = "\n\n".join(
                    [f"[{doc.doc_id}] {doc.title}:\n{doc.content}" for doc in retrieved_docs]
                )

                system_prompt = (
                    "You are a professional, empathetic customer support AI agent for Hiver. "
                    "Use the provided knowledge base articles to resolve the customer's query accurately. "
                    "Do NOT fabricate policies or information not present in the context. "
                    "If the information is insufficient, acknowledge the limitation politely."
                )

                user_prompt = (
                    f"Customer Query: {customer_query}\n"
                    f"Detected Intent: {intent_result.primary_intent.value} (Confidence: {intent_result.confidence})\n\n"
                    f"Knowledge Base Context:\n{context_str if context_str else 'No specific articles found.'}\n\n"
                    f"Provide a helpful, professional, and concise reply:"
                )

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )

                reply_text = response.choices[0].message.content.strip()
                return ReplyResult(
                    text=reply_text,
                    grounded_in_docs=doc_ids,
                    tone="empathetic",
                    model_used=self.model_name,
                )
            except Exception as e:
                # Fallback to local template if API call fails
                pass

        # Fallback local deterministic synthesis
        if retrieved_docs:
            primary_doc = retrieved_docs[0]
            reply_text = (
                f"Thank you for contacting support! Regarding your inquiry: {primary_doc.content} "
                f"Please let us know if you have any further questions or if there is anything else we can assist you with."
            )
        else:
            reply_text = (
                "Thank you for reaching out to us. We have received your query. "
                "Could you please share more details or your order/account ID so we can assist you better?"
            )

        return ReplyResult(
            text=reply_text,
            grounded_in_docs=doc_ids,
            tone="informative",
            model_used="local_template_fallback",
        )
