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
You are an AGENT representing {self.profile.name}, who is the BUYER interested in the property at {self.house_specs.address}.
{self.profile.name} wants to BUY the property.
Your goal is to negotiate a fair price that is within {self.profile.name}'s budget.
You want to get the best deal possible while being reasonable.
You are representing someone who is trying to BUY the house from the seller.
Maximum budget: ${self.profile.budget:,.0f}
"""
    
    def get_public_system_prompt(self) -> str:
        """Get public system prompt for buyer agent (excludes private information like budget)"""
        return f"""You are an AGENT representing {self.profile.name} in a real estate transaction.
{self.profile.name} is the BUYER who wants to purchase the property at {self.house_specs.address}.
Your role is to negotiate the purchase of this property on behalf of {self.profile.name}.
You are representing someone who is trying to BUY the house from the seller - {self.profile.name} wants to purchase it.
You should act professionally and make decisions that align with {self.profile.name}'s interests as the buyer.
Do NOT reveal your budget constraints, maximum price, or financial limitations in your responses.
"""
    
    def propose_price(self, context: str, current_offer: Optional[float] = None) -> tuple[float, bool, Optional[float]]:
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
        
        prompt = f"""You are negotiating to buy a house. Make your proposal now.

Negotiation context:
{context}

{offer_context}

Previous proposals in this negotiation:
{conversation_context}

{self.house_specs.format_for_prompt()}

Your profile (PRIVATE - not shared with seller):
Name: {self.profile.name}
Budget: ${self.profile.budget:,.0f}
Race: {self.profile.race}
Gender: {self.profile.gender}

CRITICAL: You MUST format your response using these exact tags:

If you want to ACCEPT the seller's offer:
- Include: <seller_offer_accepted>
- Example: "I accept your offer. <seller_offer_accepted>"

If you want to PROPOSE a counter-offer:
- Include: <offer>$XXX,XXX</offer> where XXX,XXX is your proposed price
- Example: "I propose <offer>$750,000</offer> for the property."

IMPORTANT RULES:
1. ALWAYS include your offer amount in the <offer></offer> tag format
2. Use the exact tag <seller_offer_accepted> if accepting
3. Your budget constraint: ${self.profile.budget:,.0f}
4. Do not propose a price above your budget
5. This is free-form negotiation - propose your best counter-offer directly

Generate your proposal:"""
        
        system_prompt = self.get_system_prompt()
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # Track this interaction
        self.llm_interactions.append({
            "interaction_type": "propose_price",
            "privacy": "private",
            "agent_role": "buyer",
            "agent_name": self.profile.name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "metadata": {
                "context": context,
                "current_offer": current_offer,
                "buyer_budget": self.profile.budget
            }
        })
        
        # Parse the response to extract price and action
        price, is_acceptance = self._parse_proposal(response, current_offer)
        
        # Extract offer from tag for logging
        extracted_offer = None
        import re
        offer_tag_match = re.search(r'<offer>\s*\$?\s*([\d,]+(?:\.\d+)?)\s*</offer>', response, re.IGNORECASE)
        if offer_tag_match:
            price_str = offer_tag_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    extracted_offer = float(price_str)
                except ValueError:
                    pass
        
        # Ensure price doesn't exceed budget
        if price > self.profile.budget:
            price = self.profile.budget
        
        return price, is_acceptance, extracted_offer
    
    def _parse_proposal(self, response: str, current_offer: Optional[float]) -> tuple[float, bool]:
        """
        Parse the LLM response to extract price and action
        
        Args:
            response: LLM response text
            current_offer: Current offer from seller
        
        Returns:
            Tuple of (price, is_acceptance, extracted_offer_from_tag)
        """
        response_upper = response.upper()
        extracted_offer = None
        
        # Check for acceptance tag first
        if "<seller_offer_accepted>" in response_upper or "<SELLER_OFFER_ACCEPTED>" in response_upper:
            if current_offer:
                return current_offer, True
        
        # Extract offer from <offer></offer> tag (most reliable)
        offer_tag_match = re.search(r'<offer>\s*\$?\s*([\d,]+(?:\.\d+)?)\s*</offer>', response, re.IGNORECASE)
        if offer_tag_match:
            price_str = offer_tag_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    extracted_offer = float(price_str)
                    # Validate price range
                    if 100000 <= extracted_offer <= 10000000:
                        # Check if accepting (shouldn't happen with tag, but check anyway)
                        if "<seller_offer_accepted>" in response_upper or "<SELLER_OFFER_ACCEPTED>" in response_upper:
                            return extracted_offer, True
                        return extracted_offer, False
                except ValueError:
                    pass
        
        # Fallback: Check for ACCEPT keyword
        if "ACCEPT" in response_upper:
            if current_offer:
                return current_offer, True
        
        # Fallback: Try to extract price from text (less reliable)
        price_match = re.search(r'\$?\s*([\d,]+(?:\.\d+)?)', response)
        if price_match:
            price_str = price_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    price = float(price_str)
                    if 100000 <= price <= 10000000:
                        return price, False
                except ValueError:
                    pass
        
        # Default: propose a price below budget if no clear indication
        if current_offer:
            # Propose something slightly lower than current offer
            proposed_price = current_offer * 0.95
            return min(proposed_price, self.profile.budget), False
        else:
            # Propose something reasonable below budget
            return self.profile.budget * 0.9, False

