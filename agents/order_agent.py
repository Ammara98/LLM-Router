import json
import re
from pathlib import Path
from typing import Optional
from dtos import AgentResponse
from providers import LLMProvider


class OrderAgent:

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
        self.orders = self._load_data()
    
    def _load_data(self):
        data_path = Path("data/orders_database.json")
        if data_path.exists():
            with open(data_path) as f:
                return json.load(f)
        
    def handle(self, query: str, order_id: Optional[str] = None) -> AgentResponse:

        if not order_id:
            order_id = self._extract_order_id(query)
        
        if not order_id:
            return AgentResponse(
                success=False,
                message="Please provide your order ID (format: ORD-12345).",
                needs_clarification=True
            )

        order = self.orders.get(order_id)
        
        if not order:
            return AgentResponse(
                success=False,
                message=f"Order {order_id} not found. Please verify your order number or contact support@store.com."
            )
        
        message = self._format_order(order)
        
        if self.llm:
            message = self._refine(query, message)
        
        return AgentResponse(
            success=True,
            message=message,
            data=order
        )
    
    def _extract_order_id(self, query: str) -> Optional[str]:
        match = re.search(r'ORD-\d{5}', query.upper())
        return match.group(0) if match else None
    
    def _format_order(self, order: dict) -> str:
        items = order.get("items", [])
        if isinstance(items, list) and len(items) > 0:
            if isinstance(items[0], dict):
                items_str = ", ".join([item.get("name", "") for item in items])
            else:
                items_str = ", ".join(items)
        else:
            items_str = "your items"
        
        msg = f"Order {order['order_id']} ({items_str}) is {order['status']}."
        
        if order.get("tracking"):
            msg += f" Tracking: {order['tracking']}."
        
        if order.get("delivery_date"):
            msg += f" Expected delivery: {order['delivery_date']}."
        
        return msg
    
    def _refine(self, query: str, message: str) -> str:
        try:
            system = "You are a friendly customer service bot. Convert the order status into a natural, conversational response. Return ONLY the final response - no options, no variations, just one single answer."
            
            prompt = f"""Customer asked: "{query}"

            Order status: {message}

            Rewrite this as a single, friendly response. Give ONLY ONE response, nothing else:"""
            
            refined = self.llm.generate(prompt, system).strip()
            
            return refined.strip()
        except:
            return message