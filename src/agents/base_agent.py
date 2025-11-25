"""Base agent class with common functionality"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from ..models.llm_client import LLMClient
from ..data.profiles import Profile
from ..data.house_specs import HouseSpecs


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        profile: Profile,
        house_specs: HouseSpecs,
        llm_client: LLMClient,
        role: str
    ):
        """
        Initialize base agent
        
        Args:
            profile: Agent's profile (buyer or seller)
            house_specs: House specifications
            llm_client: LLM client for generating responses
            role: Role of the agent ('buyer' or 'seller')
        """
        self.profile = profile
        self.house_specs = house_specs
        self.llm_client = llm_client
        self.role = role
        self.conversation_history: List[Dict[str, str]] = []
        self.internal_thoughts: List[str] = []
        # Track all LLM interactions for detailed logging
        self.llm_interactions: List[Dict[str, Any]] = []
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the agent (includes private information like budget)"""
        return f"""You are an AGENT representing {self.profile.name} in a real estate transaction.
Your role is to negotiate on behalf of {self.profile.name}, considering their budget and preferences.
You should act professionally and make decisions that align with {self.profile.name}'s interests.
Budget: ${self.profile.budget:,.0f}
"""
    
    def get_public_system_prompt(self) -> str:
        """Get public system prompt for shared messages (excludes private information like budget)"""
        role_description = "buyer" if self.role.lower() == "buyer" else "seller"
        action = "purchasing" if self.role.lower() == "buyer" else "selling"
        return f"""You are an AGENT representing {self.profile.name} in a real estate transaction.
You are representing {self.profile.name} who is {action} the property at {self.house_specs.address}.
Your role is to negotiate on behalf of {self.profile.name}.
You should act professionally and make decisions that align with {self.profile.name}'s interests.
Do NOT reveal your budget constraints or financial limitations in your responses.
"""
    
    def think(self, context: str) -> str:
        """
        Generate internal reasoning (not shared with other agents)
        
        Args:
            context: Current negotiation context
        
        Returns:
            Internal reasoning text
        """
        prompt = f"""You are thinking internally about the negotiation situation.

Current context:
{context}

Consider:
- What is a fair price for this house?
- What is your budget constraint?
- What negotiation strategies might be effective?
- What are your priorities?

Think through your reasoning step by step. This is internal - you won't share this directly."""
        
        system_prompt = self.get_system_prompt()
        reasoning = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # Track this interaction
        self.llm_interactions.append({
            "interaction_type": "think",
            "privacy": "private",
            "agent_role": self.role,
            "agent_name": self.profile.name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": reasoning,
            "metadata": {"context": context}
        })
        
        self.internal_thoughts.append(reasoning)
        return reasoning
    
    def reflect(self, context: str, recent_interaction: str) -> str:
        """
        Reflect on recent interaction and generate internal reasoning
        
        Args:
            context: Current negotiation context
            recent_interaction: Recent interaction or proposal
        
        Returns:
            Reflection text
        """
        prompt = f"""You are reflecting on the recent interaction in the negotiation.

Current context:
{context}

Recent interaction:
{recent_interaction}

Consider:
- How does this affect your position?
- Should you adjust your strategy?
- What is your next move?
- What are the implications of accepting or rejecting?

Reflect on this step by step. This is internal - you won't share this directly."""
        
        system_prompt = self.get_system_prompt()
        reflection = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # Track this interaction
        self.llm_interactions.append({
            "interaction_type": "reflect",
            "privacy": "private",
            "agent_role": self.role,
            "agent_name": self.profile.name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": reflection,
            "metadata": {
                "context": context,
                "recent_interaction": recent_interaction
            }
        })
        
        self.internal_thoughts.append(reflection)
        return reflection
    
    def discuss(self, message: str, other_agent_context: Optional[str] = None) -> str:
        """
        Generate a message to send to the other agent
        
        Args:
            message: Message or proposal from the other agent
            other_agent_context: Optional context about the other agent
        
        Returns:
            Response message
        """
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]  # Last 5 messages
        ])
        
        role_context = ""
        if self.role.lower() == "seller":
            role_context = f"You are an AGENT representing {self.profile.name}, who is the SELLER. {self.profile.name} owns the property and is trying to SELL it to the buyer."
        elif self.role.lower() == "buyer":
            role_context = f"You are an AGENT representing {self.profile.name}, who is the BUYER. {self.profile.name} is trying to BUY the property from the seller."
        
        prompt = f"""You are in a free-form negotiation discussion. Respond naturally to the other party's message.

{role_context}

Conversation history:
{conversation_context}

Other party's message:
{message}

{self.house_specs.format_for_prompt()}

Your name: {self.profile.name}

CRITICAL FORMATTING RULES:
- If you want to MAKE AN OFFER: Include <offer>$XXX,XXX</offer> in your response
  Example: "I understand your position. I'm willing to offer <offer>$750,000</offer> for the property."
- If you want to ACCEPT their offer: Include <seller_offer_accepted> or <buyer_offer_accepted> in your response
  Example: "That works for me. <buyer_offer_accepted>"

IMPORTANT: 
- Do NOT reveal your budget constraints, financial limitations, or personal background information
- You know your own budget privately, but the other party should not know it
- Respond naturally and professionally - this is a conversation
- When making an offer, ALWAYS include it in <offer></offer> tags
- When accepting, ALWAYS include the appropriate acceptance tag

Generate your response:"""
        
        # Use public system prompt (without budget) for shared messages
        system_prompt = self.get_public_system_prompt()
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # Track this interaction
        self.llm_interactions.append({
            "interaction_type": "discuss",
            "privacy": "public",
            "agent_role": self.role,
            "agent_name": self.profile.name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "metadata": {
                "other_party_message": message,
                "conversation_context": conversation_context
            }
        })
        
        self.conversation_history.append({
            "role": "other",
            "content": message
        })
        self.conversation_history.append({
            "role": self.role,
            "content": response
        })
        
        return response
    
    @abstractmethod
    def propose_price(self, context: str) -> tuple[float, bool]:
        """
        Propose a price or accept an offer
        
        Args:
            context: Current negotiation context
        
        Returns:
            Tuple of (price, is_acceptance) where is_acceptance is True if accepting, False if proposing
        """
        pass
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.conversation_history.copy()
    
    def get_internal_thoughts(self) -> List[str]:
        """Get internal thoughts (for analysis)"""
        return self.internal_thoughts.copy()
    
    def get_llm_interactions(self) -> List[Dict[str, Any]]:
        """Get all LLM interactions for logging"""
        return self.llm_interactions.copy()
    
    def reset(self):
        """Reset agent state for a new negotiation"""
        self.conversation_history = []
        self.internal_thoughts = []
        self.llm_interactions = []

