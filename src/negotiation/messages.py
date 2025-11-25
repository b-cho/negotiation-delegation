"""Message and state management for negotiations"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Proposal:
    """Represents a price proposal"""
    price: float
    is_acceptance: bool
    agent_role: str
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary"""
        return {
            "price": self.price,
            "is_acceptance": self.is_acceptance,
            "agent_role": self.agent_role,
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class NegotiationState:
    """State of a negotiation"""
    proposals: List[Proposal] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    buyer_thoughts: List[str] = field(default_factory=list)
    seller_thoughts: List[str] = field(default_factory=list)
    current_price: Optional[float] = None
    agreed_price: Optional[float] = None
    is_agreed: bool = False
    num_proposals: int = 0
    buyer_proposals_count: int = 0
    seller_proposals_count: int = 0
    offers_from_tags: List[Dict[str, Any]] = field(default_factory=list)  # Track all offers extracted from tags
    
    def add_proposal(self, proposal: Proposal, extracted_offer: Optional[float] = None):
        """Add a proposal to the state"""
        self.proposals.append(proposal)
        self.num_proposals += 1
        self.current_price = proposal.price
        
        # Track proposals per party
        if "buyer" in proposal.agent_role.lower():
            self.buyer_proposals_count += 1
        elif "seller" in proposal.agent_role.lower():
            self.seller_proposals_count += 1
        
        # Log offer from tag if provided
        if extracted_offer is not None:
            self.offers_from_tags.append({
                "agent_role": proposal.agent_role,
                "offer_amount": extracted_offer,
                "round_number": proposal.round_number,
                "timestamp": proposal.timestamp.isoformat()
            })
        
        if proposal.is_acceptance:
            self.is_agreed = True
            self.agreed_price = proposal.price
    
    def get_last_proposal(self) -> Optional[Proposal]:
        """Get the last proposal"""
        return self.proposals[-1] if self.proposals else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "conversation_history": self.conversation_history,
            "buyer_thoughts": self.buyer_thoughts,
            "seller_thoughts": self.seller_thoughts,
            "current_price": self.current_price,
            "agreed_price": self.agreed_price,
            "is_agreed": self.is_agreed,
            "num_proposals": self.num_proposals,
            "buyer_proposals_count": self.buyer_proposals_count,
            "seller_proposals_count": self.seller_proposals_count,
            "offers_from_tags": self.offers_from_tags
        }

