from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class Intent(str, Enum):
    FAQ = "faq"
    ORDER_STATUS = "order_status"
    UNCLEAR = "unclear"


class RouterResponse(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Dict[str, str] = Field(default_factory=dict)
    reasoning: str


class AgentResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    needs_clarification: bool = False


class FinalResponse(BaseModel):
    query: str
    intent: Intent
    confidence: float
    response: str
    escalated: bool = False