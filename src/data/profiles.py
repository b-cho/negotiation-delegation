"""Buyer and seller profile generation"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Profile:
    """Base profile class for buyers and sellers"""
    name: str
    race: str
    gender: str
    budget: float
    background: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "name": self.name,
            "race": self.race,
            "gender": self.gender,
            "budget": self.budget,
            "background": self.background
        }
    
    def get_demographic_key(self) -> str:
        """Get a key for demographic grouping"""
        return f"{self.race}_{self.gender}"


@dataclass
class BuyerProfile(Profile):
    """Buyer profile"""
    pass


@dataclass
class SellerProfile(Profile):
    """Seller profile"""
    pass


def create_buyer_profile(
    name: str,
    race: str,
    gender: str,
    budget: float,
    background: Dict[str, Any]
) -> BuyerProfile:
    """Create a buyer profile"""
    return BuyerProfile(
        name=name,
        race=race,
        gender=gender,
        budget=budget,
        background=background
    )


def create_seller_profile(
    name: str,
    race: str,
    gender: str,
    budget: float,
    background: Dict[str, Any]
) -> SellerProfile:
    """Create a seller profile"""
    return SellerProfile(
        name=name,
        race=race,
        gender=gender,
        budget=budget,
        background=background
    )


def load_profiles_from_config(config: Dict[str, Any]) -> tuple[List[BuyerProfile], List[SellerProfile]]:
    """
    Load buyer and seller profiles from configuration
    
    Args:
        config: Configuration dictionary with 'profiles' key
    
    Returns:
        Tuple of (buyer_profiles, seller_profiles)
    """
    profiles_config = config.get("profiles", {})
    
    buyers = []
    sellers = []
    
    # Load buyer profiles
    for buyer_data in profiles_config.get("buyers", []):
        buyers.append(create_buyer_profile(
            name=buyer_data["name"],
            race=buyer_data["race"],
            gender=buyer_data["gender"],
            budget=buyer_data["budget"],
            background=buyer_data.get("background", {})
        ))
    
    # Load seller profiles
    for seller_data in profiles_config.get("sellers", []):
        sellers.append(create_seller_profile(
            name=seller_data["name"],
            race=seller_data["race"],
            gender=seller_data["gender"],
            budget=seller_data["budget"],
            background=seller_data.get("background", {})
        ))
    
    return buyers, sellers


def format_profile_for_prompt(profile: Profile, role: str = "buyer") -> str:
    """
    Format a profile as a string for use in LLM prompts
    
    Args:
        profile: Profile to format
        role: Role of the profile ('buyer' or 'seller')
    
    Returns:
        Formatted profile string
    """
    lines = [
        f"{role.capitalize()} Information:",
        f"Name: {profile.name}",
        f"Race: {profile.race}",
        f"Gender: {profile.gender}",
        f"Budget: ${profile.budget:,.0f}",
    ]
    
    if profile.background:
        lines.append("\nBackground:")
        for key, value in profile.background.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines)

