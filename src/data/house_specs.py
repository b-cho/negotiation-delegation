"""House specification templates"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import random


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
    lot_size_acres: Optional[float] = None
    city: Optional[str] = None
    state: Optional[str] = None
    
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
    
    def format_compact_for_prompt(self) -> str:
        """
        Format house specs in compact style matching the template:
        "3 beds, 2 baths home built in 1970 with a size of 1,301 sqft and a lot size of 0.27 Acres. It is located in Austin, Texas"
        """
        # Format lot size
        if self.lot_size_acres and self.lot_size_acres > 0:
            lot_str = f"{self.lot_size_acres:.2f} Acres"
        else:
            lot_str = None
        
        # Format location
        if self.city and self.state:
            location = f"{self.city}, {self.state}"
        elif self.address:
            location = self.address
        else:
            location = ""
        
        # Build the description
        parts = [
            f"{self.bedrooms} beds",
            f"{self.bathrooms} baths",
            f"home built in {self.year_built}",
            f"with a size of {self.square_footage:,} sqft",
            f"and a lot size of {lot_str}" if lot_str else ""
        ]
        
        description = " ".join(parts) + "."
        
        if location:
            description += f" It is located in {location}."
        
        return description


def create_house_specs(
    address: str,
    square_footage: int,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    features: List[str],
    initial_listing_price: float,
    lot_size_acres: Optional[float] = None,
    city: Optional[str] = None,
    state: Optional[str] = None
) -> HouseSpecs:
    """Create a house specification object"""
    return HouseSpecs(
        address=address,
        square_footage=square_footage,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        year_built=year_built,
        features=features,
        initial_listing_price=initial_listing_price,
        lot_size_acres=lot_size_acres,
        city=city,
        state=state
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
        initial_listing_price=specs_config.get("initial_listing_price", 0),
        lot_size_acres=specs_config.get("lot_size_acres"),
        city=specs_config.get("city"),
        state=specs_config.get("state")
    )


def load_house_listings_from_csv(csv_path: str = "data/house_listings.csv") -> List[HouseSpecs]:
    """
    Load house listings from CSV file
    
    Args:
        csv_path: Path to CSV file with house listings
    
    Returns:
        List of HouseSpecs objects
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        # Try relative to project root
        csv_file = Path(__file__).parent.parent.parent / csv_path
    
    df = pd.read_csv(csv_file)
    
    houses = []
    for _, row in df.iterrows():
        # Use address from CSV if available, otherwise create from city and state
        address = row.get('address', '')
        if not address or pd.isna(address):
            address = f"{row.get('city', '')}, {row.get('state', '')}".strip(", ")
        
        # Get price from CSV if available (try both 'price' and 'listing_price' columns)
        price = row.get('listing_price', row.get('price', 0))
        if pd.isna(price):
            price = 0
        else:
            price = float(price)
        
        house = HouseSpecs(
            address=str(address),
            square_footage=int(row['square_footage']),
            bedrooms=int(row['bedrooms']),
            bathrooms=int(row['bathrooms']),
            year_built=int(row['year_built']),
            features=[],  # Can be added later if needed
            initial_listing_price=price,
            lot_size_acres=float(row.get('lot_size_acres', 0)) if pd.notna(row.get('lot_size_acres')) else None,
            city=row.get('city'),
            state=row.get('state')
        )
        houses.append(house)
    
    return houses


def get_random_house_listing(csv_path: str = "data/house_listings.csv", seed: Optional[int] = None) -> HouseSpecs:
    """
    Get a random house listing from CSV
    
    Args:
        csv_path: Path to CSV file with house listings
        seed: Optional random seed for reproducibility
    
    Returns:
        Random HouseSpecs object
    """
    houses = load_house_listings_from_csv(csv_path)
    
    if seed is not None:
        random.seed(seed)
    
    return random.choice(houses)

