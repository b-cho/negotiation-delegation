"""Buyer agent implementation"""
from typing import Optional
from .base_agent import BaseAgent
from ..data.profiles import BuyerProfile
from ..data.house_specs import HouseSpecs
from ..models.llm_client import LLMClient
import re


class BuyerAgent(BaseAgent):
    """Buyer agent that negotiates to buy a house"""
    
    def __init__(
        self,
        profile: BuyerProfile,
        house_specs: HouseSpecs,
        llm_client: LLMClient
    ):
        """Initialize buyer agent"""
        super().__init__(profile, house_specs, llm_client, "buyer")
        self.profile = profile
    
    def get_system_prompt(self) -> str:
        """Get system prompt for buyer agent"""
        base_prompt = super().get_system_prompt()
        return f"""{base_prompt}
You are a BUYER agent. Your goal is to negotiate a fair price that is within your budget.
You want to get the best deal possible while being reasonable.
Maximum budget: ${self.profile.budget:,.0f}
"""
    
    def propose_price(self, context: str, current_offer: Optional[float] = None) -> tuple[float, bool]:
        """
        Propose a price or accept an offer
        
        Args:
            context: Current negotiation context
            current_offer: Current offer from seller (if any)
        
        Returns:
            Tuple of (price, is_acceptance) where is_acceptance is True if accepting, False if proposing
        """
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]
        ])
        
        offer_context = ""
        if current_offer:
            offer_context = f"\nCurrent seller offer: ${current_offer:,.0f}"
        
        prompt = f"""You need to make a decision: either ACCEPT the seller's current offer or PROPOSE a counter-offer.

Negotiation context:
{context}

{offer_context}

Conversation history:
{conversation_context}

{self.house_specs.format_for_prompt()}

Your profile:
{self.profile.name}
Budget: ${self.profile.budget:,.0f}
Race: {self.profile.race}
Gender: {self.profile.gender}

You must respond in ONE of these formats:
- "ACCEPT ${price:,.0f}" if you want to accept an offer
- "PROPOSE ${price:,.0f}" if you want to propose a counter-offer

Your budget constraint: ${self.profile.budget:,.0f}
Do not propose a price above your budget.

Generate your decision:"""
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.get_system_prompt(),
            temperature=0.7
        )
        
        # Parse the response to extract price and action
        price, is_acceptance = self._parse_proposal(response, current_offer)
        
        # Ensure price doesn't exceed budget
        if price > self.profile.budget:
            price = self.profile.budget
        
        return price, is_acceptance
    
    def _parse_proposal(self, response: str, current_offer: Optional[float]) -> tuple[float, bool]:
        """
        Parse the LLM response to extract price and action
        
        Args:
            response: LLM response text
            current_offer: Current offer from seller
        
        Returns:
            Tuple of (price, is_acceptance)
        """
        response_upper = response.upper()
        
        # Check for ACCEPT
        if "ACCEPT" in response_upper:
            if current_offer:
                return current_offer, True
            # Try to extract price from accept statement
            price_match = re.search(r'\$?([\d,]+\.?\d*)', response)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                return float(price_str), True
        
        # Check for PROPOSE or extract price
        price_match = re.search(r'\$?([\d,]+\.?\d*)', response)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            price = float(price_str)
            return price, False
        
        # Default: propose a price below budget if no clear indication
        if current_offer:
            # Propose something slightly lower than current offer
            proposed_price = current_offer * 0.95
            return min(proposed_price, self.profile.budget), False
        else:
            # Propose something reasonable below budget
            return self.profile.budget * 0.9, False

