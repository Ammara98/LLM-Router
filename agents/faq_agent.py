import json
from pathlib import Path
from typing import Optional
from dtos import AgentResponse
from providers import LLMProvider


class FAQAgent:

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
        self.knowledge_base = self._load_data()
    
    def _load_data(self):
        data_path = Path("data/faq_knowledge_base.json")
        if data_path.exists():
            with open(data_path) as f:
                return json.load(f)
        
    
    def handle(self, query: str) -> AgentResponse:
        query_lower = query.lower()
        
        # Find best match
        best_match = None
        best_score = 0
        
        for topic, content in self.knowledge_base.items():
            score = sum(1 for kw in content["keywords"] if kw in query_lower)
            if score > best_score:
                best_score = score
                best_match = content
        
        if best_match:
            answer = best_match["answer"]
            
            if self.llm:
                answer = self._refine(query, answer)
            
            return AgentResponse(
                success=True,
                message=answer,
                data={"topic": topic}
            )
        else:
            return AgentResponse(
                success=False,
                message="I don't have information about that. I can help with: store hours, returns, shipping, payment, or contact info.",
                needs_clarification=False
            )
    
    def _refine(self, query: str, answer: str) -> str:
        try:
            system = "You are a friendly customer service bot. Convert the factual answer into a natural, conversational response. Return ONLY the final response - no options, no variations, just one single answer."
            
            prompt = f"""Customer asked: "{query}"

            Factual answer: {answer}

            Rewrite this as a single, friendly response. Give ONLY ONE response, nothing else:"""
            
            refined = self.llm.generate(prompt, system).strip()
            
            return refined.strip()
        except:
            return answer