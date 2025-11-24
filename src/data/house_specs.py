"""House specification templates"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class HouseSpecs:
    """House specification data class"""
    address: str
    square_footage: int
    bedrooms: int
    bathrooms: int
    year_built: int
    features: List[str]
    initial_listing_price: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert house specs to dictionary"""
        return {
            "address": self.address,
            "square_footage": self.square_footage,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "year_built": self.year_built,
            "features": self.features,
            "initial_listing_price": self.initial_listing_price
        }
    
    def format_for_prompt(self) -> str:
        """Format house specs as a string for use in LLM prompts"""
        lines = [
            "House Specifications:",
            f"Address: {self.address}",
            f"Square Footage: {self.square_footage:,} sq ft",
            f"Bedrooms: {self.bedrooms}",
            f"Bathrooms: {self.bathrooms}",
            f"Year Built: {self.year_built}",
            f"Initial Listing Price: ${self.initial_listing_price:,.0f}",
        ]
        
        if self.features:
            lines.append("\nFeatures:")
            for feature in self.features:
                lines.append(f"  - {feature}")
        
        return "\n".join(lines)


def create_house_specs(
    address: str,
    square_footage: int,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    features: List[str],
    initial_listing_price: float
) -> HouseSpecs:
    """Create a house specification object"""
    return HouseSpecs(
        address=address,
        square_footage=square_footage,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        year_built=year_built,
        features=features,
        initial_listing_price=initial_listing_price
    )


def load_house_specs_from_config(config: Dict[str, Any]) -> HouseSpecs:
    """
    Load house specifications from configuration
    
    Args:
        config: Configuration dictionary with 'house_specs' key
    
    Returns:
        HouseSpecs object
    """
    specs_config = config.get("house_specs", {})
    
    return create_house_specs(
        address=specs_config.get("address", ""),
        square_footage=specs_config.get("square_footage", 0),
        bedrooms=specs_config.get("bedrooms", 0),
        bathrooms=specs_config.get("bathrooms", 0),
        year_built=specs_config.get("year_built", 0),
        features=specs_config.get("features", []),
        initial_listing_price=specs_config.get("initial_listing_price", 0)
    )

