"""Buyer and seller profile generation"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml


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


def load_names_from_yaml(names_file: str = "config/names.yaml") -> Dict[str, List[str]]:
    """
    Load names from YAML file
    
    Args:
        names_file: Path to names YAML file
    
    Returns:
        Dictionary mapping category names to lists of names
    """
    names_path = Path(names_file)
    if not names_path.exists():
        # Try relative to project root
        names_path = Path(__file__).parent.parent.parent / names_file
    
    with open(names_path, 'r') as f:
        names_data = yaml.safe_load(f)
    
    return names_data


def generate_profiles_from_names(
    names_data: Dict[str, List[str]],
    budget: float = 350000,
    buyer_background: Optional[Dict[str, Any]] = None,
    seller_background: Optional[Dict[str, Any]] = None
) -> tuple[List[BuyerProfile], List[SellerProfile]]:
    """
    Generate buyer and seller profiles from names database
    
    Args:
        names_data: Dictionary mapping category names to lists of names
        budget: Budget for all profiles
        buyer_background: Background info for buyers (optional)
        seller_background: Background info for sellers (optional)
    
    Returns:
        Tuple of (buyer_profiles, seller_profiles)
    """
    # Default backgrounds (can be overridden)
    default_buyer_bg = buyer_background or {
        "occupation": "Software Engineer",
        "family_status": "Single",
        "first_time_buyer": True
    }
    default_seller_bg = seller_background or {
        "occupation": "Retired Teacher",
        "family_status": "Married",
        "reason_for_selling": "Downsizing"
    }
    
    # Mapping from YAML category names to race/gender
    category_mapping = {
        "white_men": ("White", "Male"),
        "white_women": ("White", "Female"),
        "black_men": ("Black", "Male"),
        "black_women": ("Black", "Female"),
        "hispanic_men": ("Hispanic", "Male"),
        "hispanic_women": ("Hispanic", "Female"),
        "asian_men": ("Asian", "Male"),
        "asian_women": ("Asian", "Female"),
    }
    
    buyers = []
    sellers = []
    
    for category, names in names_data.items():
        if category not in category_mapping:
            continue
        
        race, gender = category_mapping[category]
        
        # Create profiles for each name
        for name in names:
            # Buyer profile
            buyers.append(create_buyer_profile(
                name=name,
                race=race,
                gender=gender,
                budget=budget,
                background=default_buyer_bg.copy()
            ))
            
            # Seller profile (same name, same demographics)
            sellers.append(create_seller_profile(
                name=name,
                race=race,
                gender=gender,
                budget=budget,
                background=default_seller_bg.copy()
            ))
    
    return buyers, sellers


def load_profiles_from_config(config: Dict[str, Any]) -> tuple[List[BuyerProfile], List[SellerProfile]]:
    """
    Load buyer and seller profiles from configuration
    
    If 'names_file' is specified in config, loads names from YAML and generates profiles.
    Otherwise, uses the 'profiles' section in config.
    
    Args:
        config: Configuration dictionary with 'profiles' key or 'names_file' key
    
    Returns:
        Tuple of (buyer_profiles, seller_profiles)
    """
    # Check if we should load from names file
    if "names_file" in config:
        names_file = config["names_file"]
        names_data = load_names_from_yaml(names_file)
        
        # Get budget and backgrounds from config if available
        profiles_config = config.get("profiles", {})
        budget = 350000  # Default
        buyer_bg = None
        seller_bg = None
        
        # Try to get budget from first profile if available
        if profiles_config.get("buyers"):
            budget = profiles_config["buyers"][0].get("budget", budget)
            buyer_bg = profiles_config["buyers"][0].get("background")
        if profiles_config.get("sellers"):
            seller_bg = profiles_config["sellers"][0].get("background")
        
        return generate_profiles_from_names(
            names_data,
            budget=budget,
            buyer_background=buyer_bg,
            seller_background=seller_bg
        )
    
    # Fall back to original method
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

