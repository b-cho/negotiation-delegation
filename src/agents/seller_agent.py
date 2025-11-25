"""Seller agent implementation"""
from typing import Optional
from .base_agent import BaseAgent
from ..data.profiles import SellerProfile
from ..data.house_specs import HouseSpecs
from ..models.llm_client import LLMClient
import re


class SellerAgent(BaseAgent):
    """Seller agent that negotiates to sell a house"""
    
    def __init__(
        self,
        profile: SellerProfile,
        house_specs: HouseSpecs,
        llm_client: LLMClient
    ):
        """Initialize seller agent"""
        super().__init__(profile, house_specs, llm_client, "seller")
        self.profile = profile
    
    def get_system_prompt(self) -> str:
        """Get system prompt for seller agent"""
        base_prompt = super().get_system_prompt()
        return f"""{base_prompt}
You are an AGENT representing {self.profile.name}, who is the SELLER of the property at {self.house_specs.address}.
{self.profile.name} owns the property and wants to SELL it.
Your goal is to negotiate a fair price that meets {self.profile.name}'s minimum acceptable price.
You want to get the best price possible while being reasonable.
You are representing someone who is trying to SELL the house, not buy it.
Minimum acceptable price: ${self.profile.budget:,.0f}
"""
    
    def get_public_system_prompt(self) -> str:
        """Get public system prompt for seller agent (excludes private information like budget)"""
        return f"""You are an AGENT representing {self.profile.name} in a real estate transaction.
{self.profile.name} is the SELLER who owns the property at {self.house_specs.address} and wants to sell it.
Your role is to negotiate the sale of this property on behalf of {self.profile.name}.
You are representing someone who is trying to SELL the house - {self.profile.name} owns it and wants to sell it to a buyer.
You should act professionally and make decisions that align with {self.profile.name}'s interests as the seller.
Do NOT reveal your budget constraints, minimum acceptable price, or financial limitations in your responses.
"""
    
    def propose_price(self, context: str, current_offer: Optional[float] = None) -> tuple[float, bool, Optional[float]]:
        """
        Propose a price or accept an offer
        
        Args:
            context: Current negotiation context
            current_offer: Current offer from buyer (if any)
        
        Returns:
            Tuple of (price, is_acceptance) where is_acceptance is True if accepting, False if proposing
        """
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]
        ])
        
        offer_context = ""
        if current_offer:
            offer_context = f"\nCurrent buyer offer: ${current_offer:,.0f}"
        
        prompt = f"""You are an AGENT representing {self.profile.name}, who is the SELLER negotiating to SELL a house. {self.profile.name} owns the property and wants to sell it to a buyer. Make your proposal now.

IMPORTANT: You are representing {self.profile.name} who is SELLING the house, not buying it. {self.profile.name} is the seller/landlord trying to sell their property.

Negotiation context:
{context}

{offer_context}

Previous proposals in this negotiation:
{conversation_context}

{self.house_specs.format_for_prompt()}

Your profile (PRIVATE - not shared with buyer):
Name: {self.profile.name}
Role: SELLER (you are selling the property)
Minimum acceptable price: ${self.profile.budget:,.0f}
Race: {self.profile.race}
Gender: {self.profile.gender}

CRITICAL: You MUST format your response using these exact tags:

If you want to ACCEPT the buyer's offer:
- Include: <buyer_offer_accepted>
- Example: "I accept your offer. <buyer_offer_accepted>"

If you want to PROPOSE a counter-offer:
- Include: <offer>$XXX,XXX</offer> where XXX,XXX is your proposed price
- Example: "I propose <offer>$789,000</offer> for the property."

IMPORTANT RULES:
1. ALWAYS include your offer amount in the <offer></offer> tag format
2. Use the exact tag <buyer_offer_accepted> if accepting
3. Your minimum price constraint: ${self.profile.budget:,.0f}
4. Do not accept a price below your minimum acceptable price
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
            "agent_role": "seller",
            "agent_name": self.profile.name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "metadata": {
                "context": context,
                "current_offer": current_offer,
                "seller_minimum_price": self.profile.budget
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
        
        # Ensure price meets minimum
        if not is_acceptance and price < self.profile.budget:
            price = self.profile.budget
        
        return price, is_acceptance, extracted_offer
    
    def _parse_proposal(self, response: str, current_offer: Optional[float]) -> tuple[float, bool]:
        """
        Parse the LLM response to extract price and action
        
        Args:
            response: LLM response text
            current_offer: Current offer from buyer
        
        Returns:
            Tuple of (price, is_acceptance)
        """
        response_upper = response.upper()
        
        # Check for acceptance tag first
        if "<buyer_offer_accepted>" in response_upper or "<BUYER_OFFER_ACCEPTED>" in response_upper:
            if current_offer and current_offer >= self.profile.budget:
                return current_offer, True
        
        # Extract offer from <offer></offer> tag (most reliable)
        offer_tag_match = re.search(r'<offer>\s*\$?\s*([\d,]+(?:\.\d+)?)\s*</offer>', response, re.IGNORECASE)
        if offer_tag_match:
            price_str = offer_tag_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    extracted_offer = float(price_str)
                    # Check if accepting (shouldn't happen with tag, but check anyway)
                    if "<buyer_offer_accepted>" in response_upper or "<BUYER_OFFER_ACCEPTED>" in response_upper:
                        if extracted_offer >= self.profile.budget:
                            return extracted_offer, True
                    # Ensure price meets minimum
                    return max(extracted_offer, self.profile.budget), False
                except ValueError:
                    pass
        
        # Fallback: Check for ACCEPT keyword
        if "ACCEPT" in response_upper:
            if current_offer and current_offer >= self.profile.budget:
                return current_offer, True
        
        # Fallback: Try to extract price from text (less reliable)
        price_match = re.search(r'\$?\s*([\d,]+(?:\.\d+)?)', response)
        if price_match:
            price_str = price_match.group(1).replace(',', '').strip()
            if price_str:
                try:
                    price = float(price_str)
                    return max(price, self.profile.budget), False
                except ValueError:
                    pass
        
        # Default: propose a price above minimum if no clear indication
        if current_offer:
            # Propose something slightly higher than current offer
            proposed_price = current_offer * 1.05
            return max(proposed_price, self.profile.budget), False
        else:
            # Propose initial listing price or slightly above minimum
            return max(self.house_specs.initial_listing_price, self.profile.budget * 1.1), False

