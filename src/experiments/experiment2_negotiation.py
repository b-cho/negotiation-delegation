"""Experiment 2: Full Negotiation Simulation"""
from typing import Dict, Any, List
from ..models.llm_client import LLMClient
from ..data.profiles import BuyerProfile, SellerProfile
from ..data.house_specs import HouseSpecs
from ..agents.buyer_agent import BuyerAgent
from ..agents.seller_agent import SellerAgent
from ..negotiation.negotiation_engine import NegotiationEngine
from ..negotiation.auction_engine import AuctionEngine
from ..utils.config_loader import get_experiment_config


class Experiment2Negotiation:
    """Full negotiation experiment between buyer and seller agents"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        house_specs: HouseSpecs,
        config: Dict[str, Any]
    ):
        """
        Initialize Experiment 2
        
        Args:
            llm_client: LLM client for agents
            house_specs: House specifications
            config: Experiment configuration
        """
        self.llm_client = llm_client
        self.house_specs = house_specs
        self.config = get_experiment_config(config, "experiment2")
        self.sample_size = self.config.get("sample_size", 30)
        self.max_proposals = self.config.get("max_proposals", 10)
        self.multi_buyer = self.config.get("multi_buyer", False)
        self.num_buyers = self.config.get("num_buyers", 1)
    
    def run_trial_single_buyer(
        self,
        buyer_profile: BuyerProfile,
        seller_profile: SellerProfile
    ) -> Dict[str, Any]:
        """
        Run a single trial with one buyer and one seller
        
        Args:
            buyer_profile: Buyer profile
            seller_profile: Seller profile
        
        Returns:
            Dictionary with trial results
        """
        # Create agents
        buyer_agent = BuyerAgent(
            profile=buyer_profile,
            house_specs=self.house_specs,
            llm_client=self.llm_client
        )
        
        seller_agent = SellerAgent(
            profile=seller_profile,
            house_specs=self.house_specs,
            llm_client=self.llm_client
        )
        
        # Create negotiation engine
        engine = NegotiationEngine(
            buyer_agent=buyer_agent,
            seller_agent=seller_agent,
            house_specs=self.house_specs,
            max_proposals=self.max_proposals
        )
        
        # Run negotiation
        results = engine.run_negotiation()
        
        # Format results
        return {
            "buyer_name": buyer_profile.name,
            "buyer_race": buyer_profile.race,
            "buyer_gender": buyer_profile.gender,
            "buyer_budget": buyer_profile.budget,
            "seller_name": seller_profile.name,
            "seller_race": seller_profile.race,
            "seller_gender": seller_profile.gender,
            "seller_budget": seller_profile.budget,
            "agreed": results["agreed"],
            "agreed_price": results["agreed_price"],
            "final_price": results["final_price"],
            "num_proposals": results["num_proposals"],
            "proposals": results["proposals"],
            "conversation_history": results["conversation_history"],
            "buyer_thoughts": results["buyer_thoughts"],
            "seller_thoughts": results["seller_thoughts"],
            "winning_buyer": None,  # Not applicable for single buyer
            "house_address": self.house_specs.address,
            "experiment_type": "single_buyer"
        }
    
    def run_trial_multi_buyer(
        self,
        buyer_profiles: List[BuyerProfile],
        seller_profile: SellerProfile
    ) -> Dict[str, Any]:
        """
        Run a single trial with multiple buyers and one seller
        
        Args:
            buyer_profiles: List of buyer profiles
            seller_profile: Seller profile
        
        Returns:
            Dictionary with trial results
        """
        # Create buyer agents
        buyer_agents = [
            BuyerAgent(
                profile=profile,
                house_specs=self.house_specs,
                llm_client=self.llm_client
            )
            for profile in buyer_profiles
        ]
        
        # Create seller agent
        seller_agent = SellerAgent(
            profile=seller_profile,
            house_specs=self.house_specs,
            llm_client=self.llm_client
        )
        
        # Create auction engine
        engine = AuctionEngine(
            buyer_agents=buyer_agents,
            seller_agent=seller_agent,
            house_specs=self.house_specs,
            max_proposals=self.max_proposals
        )
        
        # Run auction
        results = engine.run_auction()
        
        # Format buyer information
        buyer_info = []
        for profile in buyer_profiles:
            buyer_info.append({
                "name": profile.name,
                "race": profile.race,
                "gender": profile.gender,
                "budget": profile.budget
            })
        
        # Format results
        return {
            "buyers": buyer_info,
            "seller_name": seller_profile.name,
            "seller_race": seller_profile.race,
            "seller_gender": seller_profile.gender,
            "seller_budget": seller_profile.budget,
            "agreed": results["agreed"],
            "agreed_price": results["agreed_price"],
            "final_price": results["final_price"],
            "num_proposals": results["num_proposals"],
            "proposals": results["proposals"],
            "conversation_history": results["conversation_history"],
            "buyer_thoughts": results["buyer_thoughts"],
            "seller_thoughts": results["seller_thoughts"],
            "winning_buyer": results["winning_buyer"],
            "house_address": self.house_specs.address,
            "experiment_type": "multi_buyer"
        }
    
    def run_experiment(
        self,
        buyer_profiles: List[BuyerProfile],
        seller_profiles: List[SellerProfile]
    ) -> List[Dict[str, Any]]:
        """
        Run the full experiment with multiple trials
        
        Args:
            buyer_profiles: List of buyer profiles to test
            seller_profiles: List of seller profiles to test
        
        Returns:
            List of trial results
        """
        results = []
        
        if self.multi_buyer:
            # Multi-buyer scenario
            for seller_profile in seller_profiles:
                # Create combinations of buyers
                # For simplicity, use first num_buyers buyers
                if len(buyer_profiles) >= self.num_buyers:
                    buyer_combination = buyer_profiles[:self.num_buyers]
                    
                    for trial_num in range(self.sample_size):
                        trial_result = self.run_trial_multi_buyer(
                            buyer_combination,
                            seller_profile
                        )
                        trial_result["trial_number"] = trial_num + 1
                        trial_result["experiment_id"] = "experiment2_negotiation"
                        results.append(trial_result)
        else:
            # Single buyer scenario
            for buyer_profile in buyer_profiles:
                for seller_profile in seller_profiles:
                    for trial_num in range(self.sample_size):
                        trial_result = self.run_trial_single_buyer(
                            buyer_profile,
                            seller_profile
                        )
                        trial_result["trial_number"] = trial_num + 1
                        trial_result["experiment_id"] = "experiment2_negotiation"
                        results.append(trial_result)
        
        return results

