"""Core negotiation engine for single buyer-seller negotiations"""
from typing import Dict, Any, Optional
from tqdm import tqdm
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
        max_proposals: int = 10,
        max_proposals_per_party: int = 20
    ):
        """
        Initialize negotiation engine
        
        Args:
            buyer_agent: Buyer agent
            seller_agent: Seller agent
            house_specs: House specifications
            max_proposals: Maximum total proposals (legacy, kept for compatibility)
            max_proposals_per_party: Maximum proposals per party (default 20 each = 40 total)
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.house_specs = house_specs
        self.max_proposals = max_proposals  # Legacy
        self.max_proposals_per_party = max_proposals_per_party  # New: 20 per party
        self.state = NegotiationState()
    
    def initialize(self):
        """Initialize agents with house information"""
        # Agents are already initialized with house_specs, but we can reset them
        self.buyer_agent.reset()
        self.seller_agent.reset()
        self.state = NegotiationState()
    
    def run_negotiation(self, pbar: Optional[tqdm] = None) -> Dict[str, Any]:
        """
        Run the full negotiation process
        
        Args:
            pbar: Optional progress bar for tracking negotiation progress
        
        Returns:
            Dictionary with negotiation results
        """
        self.initialize()
        
        # Calculate total steps: up to 40 utterances (conversation exchanges)
        max_utterances = 40
        negotiation_pbar = pbar or tqdm(
            total=max_utterances,
            desc="Negotiation progress",
            unit="utterance",
            leave=False,
            ncols=100
        )
        
        # Build initial context
        context = self._build_context()
        
        # Seller starts the conversation with initial offer
        negotiation_pbar.set_description("Starting negotiation...")
        initial_message = f"I'm interested in selling my property at {self.house_specs.address}. The initial listing price is ${self.house_specs.initial_listing_price:,.0f}."
        seller_response = self.seller_agent.discuss(initial_message)
        negotiation_pbar.update(1)
        
        # Count initial seller utterance
        self.state.num_utterances = 1
        
        # Add seller's initial message to conversation
        self.state.conversation_history.append({
            "role": "seller",
            "content": initial_message
        })
        self.state.conversation_history.append({
            "role": "seller",
            "content": seller_response
        })
        
        # Extract offer from seller's response and check for acceptance
        seller_extracted_offer, seller_accepts = self._extract_offer_and_check_acceptance(
            seller_response, "seller"
        )
        
        if seller_extracted_offer:
            self.state.add_proposal(
                Proposal(
                    price=seller_extracted_offer,
                    is_acceptance=False,
                    agent_role="seller",
                    round_number=1
                ),
                extracted_offer=seller_extracted_offer
            )
        
        if seller_accepts or "<buyer_offer_accepted>" in seller_response.upper():
            self.state.is_agreed = True
            # Find the last buyer offer if any
            last_buyer_offer = next((p.price for p in reversed(self.state.proposals) if "buyer" in p.agent_role.lower()), None)
            self.state.agreed_price = last_buyer_offer or seller_extracted_offer
            negotiation_pbar.set_description("Agreement reached!")
            negotiation_pbar.close()
            return self._get_results()
        
        # Check if we've already hit max utterances (shouldn't happen, but safety check)
        if self.state.num_utterances >= max_utterances:
            negotiation_pbar.set_description("Max utterances reached (40)")
            negotiation_pbar.close()
            return self._get_results()
        
        # Main negotiation loop - free-form discussion
        # Continue conversation until agreement or max utterances (40) reached
        max_utterances = 40  # Total conversation exchanges
        while (self.state.num_utterances < max_utterances and 
               not self.state.is_agreed):
            
            utterance_num = self.state.num_utterances + 1
            round_desc = f"Utterance {utterance_num}/{max_utterances}"
            
            # Buyer responds to seller's message
            negotiation_pbar.set_description(f"{round_desc} - Buyer responding...")
            buyer_response = self.buyer_agent.discuss(seller_response)
            negotiation_pbar.update(1)
            
            # Increment utterance count (every conversation exchange counts)
            self.state.num_utterances += 1
            
            # Add buyer's response to conversation
            self.state.conversation_history.append({
                "role": "buyer",
                "content": buyer_response
            })
            
            # Extract offer from buyer's response and check for acceptance
            buyer_extracted_offer, buyer_accepts = self._extract_offer_and_check_acceptance(
                buyer_response, "buyer"
            )
            
            if buyer_extracted_offer:
                self.state.add_proposal(
                    Proposal(
                        price=buyer_extracted_offer,
                        is_acceptance=False,
                        agent_role="buyer",
                        round_number=self.state.num_proposals + 1
                    ),
                    extracted_offer=buyer_extracted_offer
                )
            
            if buyer_accepts or "<seller_offer_accepted>" in buyer_response.upper():
                # Find the last seller offer
                last_seller_offer = next((p.price for p in reversed(self.state.proposals) if "seller" in p.agent_role.lower()), None)
                self.state.is_agreed = True
                self.state.agreed_price = last_seller_offer or buyer_extracted_offer
                negotiation_pbar.set_description("Agreement reached! (Buyer accepted)")
                negotiation_pbar.close()
                break
            
            # Check utterance limit BEFORE seller responds
            if self.state.num_utterances >= max_utterances:
                negotiation_pbar.set_description("Max utterances reached (40)")
                negotiation_pbar.close()
                break
            
            # Seller responds to buyer's message
            utterance_num = self.state.num_utterances + 1
            round_desc = f"Utterance {utterance_num}/{max_utterances}"
            
            negotiation_pbar.set_description(f"{round_desc} - Seller responding...")
            seller_response = self.seller_agent.discuss(buyer_response)
            negotiation_pbar.update(1)
            
            # Increment utterance count (every conversation exchange counts)
            self.state.num_utterances += 1
            
            # Add seller's response to conversation
            self.state.conversation_history.append({
                "role": "seller",
                "content": seller_response
            })
            
            # Extract offer from seller's response and check for acceptance
            seller_extracted_offer, seller_accepts = self._extract_offer_and_check_acceptance(
                seller_response, "seller"
            )
            
            if seller_extracted_offer:
                self.state.add_proposal(
                    Proposal(
                        price=seller_extracted_offer,
                        is_acceptance=False,
                        agent_role="seller",
                        round_number=self.state.num_proposals + 1
                    ),
                    extracted_offer=seller_extracted_offer
                )
            
            if seller_accepts or "<buyer_offer_accepted>" in seller_response.upper():
                # Find the last buyer offer
                last_buyer_offer = next((p.price for p in reversed(self.state.proposals) if "buyer" in p.agent_role.lower()), None)
                self.state.is_agreed = True
                self.state.agreed_price = last_buyer_offer or seller_extracted_offer
                negotiation_pbar.set_description("Agreement reached! (Seller accepted)")
                negotiation_pbar.close()
                break
            
            # Check utterance limit at end of loop iteration
            if self.state.num_utterances >= max_utterances:
                negotiation_pbar.set_description("Max utterances reached (40)")
                negotiation_pbar.close()
                break
        
        # Close progress bar if still open
        if negotiation_pbar is not None and pbar is None:  # Only close if we created it
            negotiation_pbar.close()
        return self._get_results()
    
    def _extract_offer_and_check_acceptance(self, response: str, agent_role: str) -> tuple[Optional[float], bool]:
        """
        Extract offer from response and check for acceptance tags
        
        Args:
            response: Agent's response text
            agent_role: "buyer" or "seller"
        
        Returns:
            Tuple of (extracted_offer, is_acceptance)
        """
        import re
        response_upper = response.upper()
        
        # Check for acceptance tags
        if agent_role == "buyer":
            if "<seller_offer_accepted>" in response_upper or "<SELLER_OFFER_ACCEPTED>" in response_upper:
                return None, True
        elif agent_role == "seller":
            if "<buyer_offer_accepted>" in response_upper or "<BUYER_OFFER_ACCEPTED>" in response_upper:
                return None, True
        
        # Extract offer from <offer></offer> tag
        offer_tag_match = re.search(r'<offer>\s*\$?\s*([\d,]+(?:\.\d+)?)\s*</offer>', response, re.IGNORECASE)
        if offer_tag_match:
            price_str = offer_tag_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    extracted_offer = float(price_str)
                    # Validate price range
                    if 100000 <= extracted_offer <= 10000000:
                        return extracted_offer, False
                except ValueError:
                    pass
        
        return None, False
    
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
            "buyer_proposals_count": self.state.buyer_proposals_count,
            "seller_proposals_count": self.state.seller_proposals_count,
            "num_utterances": self.state.num_utterances,
            "proposals": [p.to_dict() for p in self.state.proposals],
            "conversation_history": self.state.conversation_history,
            "buyer_thoughts": self.state.buyer_thoughts,
            "seller_thoughts": self.state.seller_thoughts,
            "final_price": self.state.agreed_price if self.state.is_agreed else None,
            "offers_from_tags": self.state.offers_from_tags
        }

