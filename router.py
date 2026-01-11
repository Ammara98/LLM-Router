from typing import Dict
from dtos import Intent, RouterResponse, FinalResponse
from providers import LLMProvider
from agents import FAQAgent, OrderAgent


class LLMRouter:
    SYSTEM_PROMPT = """You are a customer service intent classifier.

Classify queries into:
- FAQ: Questions about store hours, returns, shipping, payment, contact
- ORDER_STATUS: Order tracking questions (look for order IDs like ORD-XXXXX)
- UNCLEAR: Vague or ambiguous queries

Extract entities:
- order_id: Order number if present (ORD-XXXXX)

Provide:
- intent: The classification
- confidence: 0.0 to 1.0
- entities: Extracted data
- reasoning: Why you chose this"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.faq_agent = FAQAgent(llm)
        self.order_agent = OrderAgent(llm)
        self.unclear_count: Dict[str, int] = {}

    def route(self, query: str, session_id: str = "default") -> FinalResponse:
        # Step 1: Classify intent
        try:
            classification = self.llm.generate_json(
                prompt=query,
                system_prompt=self.SYSTEM_PROMPT,
                model_class=RouterResponse
            )
        except Exception as e:
            print(f"Classification error: {e}")
            return self._escalate(query)

        # Step 2: Handle unclear
        if classification.intent == Intent.UNCLEAR or classification.confidence < 0.7:
            return self._handle_unclear(query, classification, session_id)

        # Step 3: Route to agent
        if classification.intent == Intent.FAQ:
            agent_response = self.faq_agent.handle(query)
        elif classification.intent == Intent.ORDER_STATUS:
            order_id = classification.entities.get("order_id")
            agent_response = self.order_agent.handle(query, order_id)
        else:
            return self._escalate(query)

        # Step 4: Handle agent response
        if agent_response.needs_clarification:
            return FinalResponse(
                query=query,
                intent=classification.intent,
                confidence=classification.confidence,
                response=agent_response.message
            )

        if not agent_response.success:
            # Intent correct but no data
            if classification.intent == Intent.FAQ:
                msg = "I don't have that info. I can help with: hours, returns, shipping, payment, contact."
            else:
                msg = agent_response.message
            return FinalResponse(
                query=query,
                intent=classification.intent,
                confidence=classification.confidence,
                response=msg
            )

        # Success
        return FinalResponse(
            query=query,
            intent=classification.intent,
            confidence=classification.confidence,
            response=agent_response.message
        )

    def _handle_unclear(self, query: str, classification: RouterResponse, session_id: str) -> FinalResponse:
        """Handle unclear queries with fallback"""
        count = self.unclear_count.get(session_id, 0)
        
        if count == 0:
            # First time - ask for clarification
            self.unclear_count[session_id] = 1
            return FinalResponse(
                query=query,
                intent=Intent.UNCLEAR,
                confidence=classification.confidence,
                response="Could you clarify? Are you asking about:\n• Store policies (hours, returns, shipping)?\n• Order status (provide order ID: ORD-XXXXX)?"
            )
        else:
            # Second time - escalate
            return self._escalate(query)

    def _escalate(self, query: str) -> FinalResponse:
        """Escalate to human support"""
        return FinalResponse(
            query=query,
            intent=Intent.UNCLEAR,
            confidence=0.0,
            response="Let me connect you with support:\n• Email: support@store.com\n• Phone: 1-800-SHOP-NOW",
            escalated=True
        )

    def reset_session(self, session_id: str):
        """Reset session history"""
        if session_id in self.unclear_count:
            del self.unclear_count[session_id]