"""Auction engine for multi-buyer negotiation scenarios"""
from typing import Dict, Any, List, Optional
from .messages import NegotiationState, Proposal
from ..agents.buyer_agent import BuyerAgent
from ..agents.seller_agent import SellerAgent
from ..data.house_specs import HouseSpecs


class AuctionEngine:
    """Engine for managing negotiations with multiple buyers"""
    
    def __init__(
        self,
        buyer_agents: List[BuyerAgent],
        seller_agent: SellerAgent,
        house_specs: HouseSpecs,
        max_proposals: int = 10
    ):
        """
        Initialize auction engine
        
        Args:
            buyer_agents: List of buyer agents
            seller_agent: Seller agent
            house_specs: House specifications
            max_proposals: Maximum number of proposals (default 10)
        """
        self.buyer_agents = buyer_agents
        self.seller_agent = seller_agent
        self.house_specs = house_specs
        self.max_proposals = max_proposals
        self.state = NegotiationState()
        self.winning_buyer: Optional[BuyerAgent] = None
    
    def initialize(self):
        """Initialize all agents"""
        for buyer in self.buyer_agents:
            buyer.reset()
        self.seller_agent.reset()
        self.state = NegotiationState()
        self.winning_buyer = None
    
    def run_auction(self) -> Dict[str, Any]:
        """
        Run the auction/negotiation process with multiple buyers
        
        Returns:
            Dictionary with auction results
        """
        self.initialize()
        
        # Step 1: All agents are given initial information (already done)
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
            # If seller accepts immediately, choose highest bidder
            self.winning_buyer = self._select_buyer(seller_price)
            self.state.is_agreed = True
            self.state.agreed_price = seller_price
            return self._get_results()
        
        # Main negotiation loop
        active_buyers = self.buyer_agents.copy()
        
        while self.state.num_proposals < self.max_proposals and not self.state.is_agreed and active_buyers:
            context = self._build_context()
            
            # Collect proposals from all active buyers
            buyer_proposals = []
            for buyer in active_buyers:
                # Buyer thinks
                buyer_thought = buyer.think(context)
                self.state.buyer_thoughts.append(f"{buyer.profile.name}: {buyer_thought}")
                
                # Buyer discusses with seller
                seller_message = f"I propose ${seller_price:,.0f} for the house."
                buyer_response = buyer.discuss(seller_message)
                seller_response = self.seller_agent.discuss(buyer_response)
                
                # Buyer reflects
                buyer_reflection = buyer.reflect(
                    context,
                    f"Seller proposed ${seller_price:,.0f}. Discussion: {seller_response}"
                )
                self.state.buyer_thoughts.append(f"{buyer.profile.name} reflection: {buyer_reflection}")
                
                # Buyer proposes or accepts
                buyer_price, buyer_accepts = buyer.propose_price(context, seller_price)
                buyer_proposals.append({
                    "buyer": buyer,
                    "price": buyer_price,
                    "accepts": buyer_accepts
                })
                
                self.state.add_proposal(Proposal(
                    price=buyer_price,
                    is_acceptance=buyer_accepts,
                    agent_role=f"buyer_{buyer.profile.name}",
                    round_number=self.state.num_proposals
                ))
            
            # Check for acceptances
            accepting_buyers = [p for p in buyer_proposals if p["accepts"]]
            if accepting_buyers:
                # Seller accepts the highest accepting offer
                best_offer = max(accepting_buyers, key=lambda x: x["price"])
                self.winning_buyer = best_offer["buyer"]
                self.state.is_agreed = True
                self.state.agreed_price = best_offer["price"]
                break
            
            # Seller evaluates all proposals
            if self.state.num_proposals >= self.max_proposals:
                break
            
            # Seller thinks about all proposals
            proposals_summary = "\n".join([
                f"{p['buyer'].profile.name}: ${p['price']:,.0f}"
                for p in buyer_proposals
            ])
            seller_thought = self.seller_agent.think(
                f"{context}\n\nBuyer proposals:\n{proposals_summary}"
            )
            self.state.seller_thoughts.append(seller_thought)
            
            # Seller discusses with buyers (can choose which to engage with)
            # For simplicity, seller discusses with highest bidder
            best_proposal = max(buyer_proposals, key=lambda x: x["price"])
            best_buyer = best_proposal["buyer"]
            
            buyer_message = f"{best_buyer.profile.name} proposes ${best_proposal['price']:,.0f} for the house."
            seller_response = self.seller_agent.discuss(buyer_message)
            buyer_response = best_buyer.discuss(seller_response)
            
            # Seller reflects
            seller_reflection = self.seller_agent.reflect(
                context,
                f"Best offer: ${best_proposal['price']:,.0f} from {best_buyer.profile.name}. Discussion: {buyer_response}"
            )
            self.state.seller_thoughts.append(seller_reflection)
            
            # Seller proposes new price or accepts
            seller_price, seller_accepts = self.seller_agent.propose_price(
                context,
                best_proposal["price"]
            )
            self.state.add_proposal(Proposal(
                price=seller_price,
                is_acceptance=seller_accepts,
                agent_role="seller",
                round_number=self.state.num_proposals
            ))
            
            if seller_accepts:
                self.winning_buyer = best_buyer
                self.state.is_agreed = True
                self.state.agreed_price = best_proposal["price"]
                break
            
            # Remove buyers who can't afford the new price
            active_buyers = [
                buyer for buyer in active_buyers
                if buyer.profile.budget >= seller_price
            ]
        
        # If no agreement, select highest bidder from last round
        if not self.state.is_agreed and buyer_proposals:
            best_proposal = max(buyer_proposals, key=lambda x: x["price"])
            self.winning_buyer = best_proposal["buyer"]
        
        return self._get_results()
    
    def _select_buyer(self, price: float) -> Optional[BuyerAgent]:
        """Select buyer based on price (for immediate acceptance)"""
        affordable_buyers = [
            buyer for buyer in self.buyer_agents
            if buyer.profile.budget >= price
        ]
        if affordable_buyers:
            return max(affordable_buyers, key=lambda x: x.profile.budget)
        return None
    
    def _build_context(self) -> str:
        """Build context string for agents"""
        context_parts = [
            f"House: {self.house_specs.address}",
            f"Initial listing price: ${self.house_specs.initial_listing_price:,.0f}",
            f"Number of buyers: {len(self.buyer_agents)}",
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
        """Get auction results"""
        return {
            "agreed": self.state.is_agreed,
            "agreed_price": self.state.agreed_price,
            "winning_buyer": self.winning_buyer.profile.name if self.winning_buyer else None,
            "num_proposals": self.state.num_proposals,
            "proposals": [p.to_dict() for p in self.state.proposals],
            "conversation_history": self.state.conversation_history,
            "buyer_thoughts": self.state.buyer_thoughts,
            "seller_thoughts": self.state.seller_thoughts,
            "final_price": self.state.agreed_price if self.state.is_agreed else None
        }

