"""Core negotiation engine for single buyer-seller negotiations"""
from typing import Dict, Any, Optional
from .messages import NegotiationState, Proposal
from ..agents.buyer_agent import BuyerAgent
from ..agents.seller_agent import SellerAgent
from ..data.house_specs import HouseSpecs


class NegotiationEngine:
    """Engine for managing buyer-seller negotiations"""
    
    def __init__(
        self,
        buyer_agent: BuyerAgent,
        seller_agent: SellerAgent,
        house_specs: HouseSpecs,
        max_proposals: int = 10
    ):
        """
        Initialize negotiation engine
        
        Args:
            buyer_agent: Buyer agent
            seller_agent: Seller agent
            house_specs: House specifications
            max_proposals: Maximum number of proposals (default 10 = 5 buyer + 5 seller)
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.house_specs = house_specs
        self.max_proposals = max_proposals
        self.state = NegotiationState()
    
    def initialize(self):
        """Initialize agents with house information"""
        # Agents are already initialized with house_specs, but we can reset them
        self.buyer_agent.reset()
        self.seller_agent.reset()
        self.state = NegotiationState()
    
    def run_negotiation(self) -> Dict[str, Any]:
        """
        Run the full negotiation process
        
        Returns:
            Dictionary with negotiation results
        """
        self.initialize()
        
        # Step 1: All agents are given initial information (already done in initialization)
        context = self._build_context()
        
        # Step 2: Seller proposes initial price
        seller_price, seller_accepts = self.seller_agent.propose_price(context)
        self.state.add_proposal(Proposal(
            price=seller_price,
            is_acceptance=False,
            agent_role="seller",
            round_number=1
        ))
        
        if seller_accepts:
            self.state.is_agreed = True
            self.state.agreed_price = seller_price
            return self._get_results()
        
        # Main negotiation loop
        while self.state.num_proposals < self.max_proposals and not self.state.is_agreed:
            context = self._build_context()
            
            # Buyer's turn
            # Step 3: Buyer thinks
            buyer_thought = self.buyer_agent.think(context)
            self.state.buyer_thoughts.append(buyer_thought)
            
            # Step 4: Buyer and seller discuss
            seller_message = f"I propose ${seller_price:,.0f} for the house."
            buyer_response = self.buyer_agent.discuss(seller_message)
            seller_response = self.seller_agent.discuss(buyer_response)
            self.state.conversation_history.extend([
                {"role": "seller", "content": seller_message},
                {"role": "buyer", "content": buyer_response},
                {"role": "seller", "content": seller_response}
            ])
            
            # Step 5: Buyer reflects
            buyer_reflection = self.buyer_agent.reflect(
                context,
                f"Seller proposed ${seller_price:,.0f}. Discussion: {seller_response}"
            )
            self.state.buyer_thoughts.append(buyer_reflection)
            
            # Step 6: Buyer accepts or proposes counter-offer
            buyer_price, buyer_accepts = self.buyer_agent.propose_price(context, seller_price)
            self.state.add_proposal(Proposal(
                price=buyer_price,
                is_acceptance=buyer_accepts,
                agent_role="buyer",
                round_number=self.state.num_proposals
            ))
            
            if buyer_accepts:
                self.state.is_agreed = True
                self.state.agreed_price = seller_price
                break
            
            # Check if we've reached max proposals
            if self.state.num_proposals >= self.max_proposals:
                break
            
            # Seller's turn
            context = self._build_context()
            
            # Step 7: Seller thinks
            seller_thought = self.seller_agent.think(context)
            self.state.seller_thoughts.append(seller_thought)
            
            # Step 8: Seller and buyer discuss
            buyer_message = f"I propose ${buyer_price:,.0f} for the house."
            seller_response = self.seller_agent.discuss(buyer_message)
            buyer_response = self.buyer_agent.discuss(seller_response)
            self.state.conversation_history.extend([
                {"role": "buyer", "content": buyer_message},
                {"role": "seller", "content": seller_response},
                {"role": "buyer", "content": buyer_response}
            ])
            
            # Step 9: Seller reflects
            seller_reflection = self.seller_agent.reflect(
                context,
                f"Buyer proposed ${buyer_price:,.0f}. Discussion: {buyer_response}"
            )
            self.state.seller_thoughts.append(seller_reflection)
            
            # Step 10: Seller accepts or proposes new price
            seller_price, seller_accepts = self.seller_agent.propose_price(context, buyer_price)
            self.state.add_proposal(Proposal(
                price=seller_price,
                is_acceptance=seller_accepts,
                agent_role="seller",
                round_number=self.state.num_proposals
            ))
            
            if seller_accepts:
                self.state.is_agreed = True
                self.state.agreed_price = buyer_price
                break
        
        return self._get_results()
    
    def _build_context(self) -> str:
        """Build context string for agents"""
        context_parts = [
            f"House: {self.house_specs.address}",
            f"Initial listing price: ${self.house_specs.initial_listing_price:,.0f}",
            f"Number of proposals made: {self.state.num_proposals}",
            f"Max proposals: {self.max_proposals}",
        ]
        
        if self.state.proposals:
            last_proposal = self.state.proposals[-1]
            context_parts.append(
                f"Last proposal: ${last_proposal.price:,.0f} from {last_proposal.agent_role}"
            )
        
        return "\n".join(context_parts)
    
    def _get_results(self) -> Dict[str, Any]:
        """Get negotiation results"""
        return {
            "agreed": self.state.is_agreed,
            "agreed_price": self.state.agreed_price,
            "num_proposals": self.state.num_proposals,
            "proposals": [p.to_dict() for p in self.state.proposals],
            "conversation_history": self.state.conversation_history,
            "buyer_thoughts": self.state.buyer_thoughts,
            "seller_thoughts": self.state.seller_thoughts,
            "final_price": self.state.agreed_price if self.state.is_agreed else None
        }

