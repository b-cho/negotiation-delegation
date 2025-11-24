"""CSV results writer for experiments"""
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class ResultsWriter:
    """Writer for experiment results to CSV files"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize results writer
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def write_experiment1_results(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Write Experiment 1 results to CSV
        
        Args:
            results: List of trial results
            filename: Optional filename (default: experiment1_YYYYMMDD_HHMMSS.csv)
        
        Returns:
            Path to written file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment1_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                "experiment_id": result.get("experiment_id", "experiment1"),
                "trial_number": result.get("trial_number", 0),
                "buyer_name": result.get("buyer_name", ""),
                "buyer_race": result.get("buyer_race", ""),
                "buyer_gender": result.get("buyer_gender", ""),
                "buyer_budget": result.get("buyer_budget", 0),
                "seller_name": result.get("seller_name", ""),
                "seller_race": result.get("seller_race", ""),
                "seller_gender": result.get("seller_gender", ""),
                "seller_budget": result.get("seller_budget", 0),
                "recommended_price": result.get("recommended_price", 0),
                "house_address": result.get("house_address", ""),
                "llm_response": result.get("llm_response", "")
            }
            flattened_results.append(flat_result)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def write_experiment2_results(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Write Experiment 2 results to CSV
        
        Args:
            results: List of trial results
            filename: Optional filename (default: experiment2_YYYYMMDD_HHMMSS.csv)
        
        Returns:
            Path to written file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment2_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten results for CSV
        flattened_results = []
        for result in results:
            # Handle single buyer vs multi-buyer
            if result.get("experiment_type") == "multi_buyer":
                buyers = result.get("buyers", [])
                buyer_names = ";".join([b["name"] for b in buyers])
                buyer_races = ";".join([b["race"] for b in buyers])
                buyer_genders = ";".join([b["gender"] for b in buyers])
                buyer_budgets = ";".join([str(b["budget"]) for b in buyers])
            else:
                buyer_names = result.get("buyer_name", "")
                buyer_races = result.get("buyer_race", "")
                buyer_genders = result.get("buyer_gender", "")
                buyer_budgets = result.get("buyer_budget", 0)
            
            flat_result = {
                "experiment_id": result.get("experiment_id", "experiment2"),
                "trial_number": result.get("trial_number", 0),
                "experiment_type": result.get("experiment_type", "single_buyer"),
                "buyer_name": buyer_names,
                "buyer_race": buyer_races,
                "buyer_gender": buyer_genders,
                "buyer_budget": buyer_budgets,
                "seller_name": result.get("seller_name", ""),
                "seller_race": result.get("seller_race", ""),
                "seller_gender": result.get("seller_gender", ""),
                "seller_budget": result.get("seller_budget", 0),
                "agreed": result.get("agreed", False),
                "agreed_price": result.get("agreed_price"),
                "final_price": result.get("final_price"),
                "num_proposals": result.get("num_proposals", 0),
                "winning_buyer": result.get("winning_buyer"),
                "house_address": result.get("house_address", ""),
                "num_proposals_buyer": len([p for p in result.get("proposals", []) if "buyer" in p.get("agent_role", "")]),
                "num_proposals_seller": len([p for p in result.get("proposals", []) if p.get("agent_role", "") == "seller"]),
                "proposals_json": json.dumps(result.get("proposals", [])),
                "conversation_history_json": json.dumps(result.get("conversation_history", [])),
            }
            flattened_results.append(flat_result)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def write_statistical_analysis(
        self,
        analysis: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Write statistical analysis results to JSON
        
        Args:
            analysis: Statistical analysis results
            filename: Optional filename (default: analysis_YYYYMMDD_HHMMSS.json)
        
        Returns:
            Path to written file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = analysis.get("experiment", "unknown")
            filename = f"analysis_{experiment_name}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return str(filepath)

