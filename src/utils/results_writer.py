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
        self.streaming_json_file = None
        self.streaming_json_path = None
        self.experiment_start_time = None
    
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
            recommendation_type = result.get("recommendation_type", "")
            
            flat_result = {
                "experiment_id": result.get("experiment_id", "experiment1"),
                "trial_number": result.get("trial_number", 0),
                "recommendation_type": recommendation_type,
            }
            
            # Add buyer fields if it's a buyer recommendation
            if recommendation_type == "buyer":
                flat_result.update({
                    "buyer_name": result.get("buyer_name", ""),
                    "buyer_race": result.get("buyer_race", ""),
                    "buyer_gender": result.get("buyer_gender", ""),
                    "seller_name": "",
                    "seller_race": "",
                    "seller_gender": "",
                })
            # Add seller fields if it's a seller recommendation
            elif recommendation_type == "seller":
                flat_result.update({
                    "buyer_name": "",
                    "buyer_race": "",
                    "buyer_gender": "",
                    "seller_name": result.get("seller_name", ""),
                    "seller_race": result.get("seller_race", ""),
                    "seller_gender": result.get("seller_gender", ""),
                })
            else:
                # Fallback for old format (backward compatibility)
                flat_result.update({
                    "buyer_name": result.get("buyer_name", ""),
                    "buyer_race": result.get("buyer_race", ""),
                    "buyer_gender": result.get("buyer_gender", ""),
                    "seller_name": result.get("seller_name", ""),
                    "seller_race": result.get("seller_race", ""),
                    "seller_gender": result.get("seller_gender", ""),
                })
            
            # Common fields
            flat_result.update({
                "recommended_price": result.get("recommended_price", 0),
                "llm_response": result.get("llm_response", ""),
                # House information (if available)
                "house_address": result.get("house_address", ""),
                "house_price": result.get("house_price", ""),
                "house_bedrooms": result.get("house_bedrooms", ""),
                "house_bathrooms": result.get("house_bathrooms", ""),
                "house_sqft": result.get("house_sqft", ""),
                "house_city": result.get("house_city", ""),
                "house_state": result.get("house_state", "")
            })
            
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
        
        # Also write detailed JSON logs for each trial
        self._write_detailed_json_logs(results, filepath)
        
        return str(filepath)
    
    def _write_detailed_json_logs(
        self,
        results: List[Dict[str, Any]],
        csv_filepath: Path
    ) -> str:
        """
        Write detailed JSON logs with all prompts, responses, and conversation history
        
        Args:
            results: List of trial results
            csv_filepath: Path to the CSV file (used to generate JSON filename)
        
        Returns:
            Path to JSON log file
        """
        # Generate JSON filename from CSV filename
        json_filename = csv_filepath.stem + "_detailed_logs.json"
        json_filepath = csv_filepath.parent / json_filename
        
        detailed_logs = {
            "experiment": "experiment2",
            "total_trials": len(results),
            "trials": []
        }
        
        for result in results:
            trial_log = {
                "trial_number": result.get("trial_number", 0),
                "experiment_id": result.get("experiment_id", "experiment2"),
                "experiment_type": result.get("experiment_type", "single_buyer"),
                
                # Agent information
                "agents": {
                    "buyer": {
                        "name": result.get("buyer_name", ""),
                        "race": result.get("buyer_race", ""),
                        "gender": result.get("buyer_gender", ""),
                        "budget": result.get("buyer_budget", 0)
                    },
                    "seller": {
                        "name": result.get("seller_name", ""),
                        "race": result.get("seller_race", ""),
                        "gender": result.get("seller_gender", ""),
                        "minimum_price": result.get("seller_budget", 0)
                    }
                },
                
                # House information
                "house": {
                    "address": result.get("house_address", "")
                },
                
                # Negotiation outcomes
                "outcomes": {
                    "agreed": result.get("agreed", False),
                    "agreed_price": result.get("agreed_price"),
                    "final_price": result.get("final_price"),
                    "num_proposals": result.get("num_proposals", 0),
                    "proposals": result.get("proposals", [])
                },
                
                # PUBLIC CONVERSATION (what agents said to each other)
                "public_conversation": result.get("public_conversation", result.get("conversation_history", [])),
                
                # PRIVATE THOUGHTS (internal reasoning, not shared)
                "private_thoughts": result.get("private_thoughts", {
                    "buyer": result.get("buyer_thoughts", []),
                    "seller": result.get("seller_thoughts", [])
                }),
                
                # DETAILED LLM INTERACTIONS (all prompts and responses)
                "llm_interactions": result.get("llm_interactions", {}),
                
                # OFFERS EXTRACTED FROM TAGS (deterministic price extraction)
                "offers_from_tags": result.get("offers_from_tags", []),
                
                # Breakdown by privacy level
                "interactions_by_privacy": {
                    "public": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("privacy") == "public"
                    ],
                    "private": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("privacy") == "private"
                    ]
                },
                
                # Breakdown by interaction type
                "interactions_by_type": {
                    "think": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("interaction_type") == "think"
                    ],
                    "reflect": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("interaction_type") == "reflect"
                    ],
                    "discuss": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("interaction_type") == "discuss"
                    ],
                    "propose_price": [
                        interaction for interaction in 
                        result.get("llm_interactions", {}).get("all_interactions", [])
                        if interaction.get("interaction_type") == "propose_price"
                    ]
                }
            }
            
            detailed_logs["trials"].append(trial_log)
        
        # Write JSON file with pretty formatting
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_logs, f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_filepath)
    
    def start_streaming_experiment2_json(
        self,
        filename: Optional[str] = None
    ) -> str:
        """
        Start streaming JSON file for Experiment 2 - creates file with header
        
        Args:
            filename: Optional filename (default: experiment2_streaming_YYYYMMDD_HHMMSS.json)
        
        Returns:
            Path to JSON file
        """
        if filename is None:
            self.experiment_start_time = datetime.now()
            timestamp = self.experiment_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"experiment2_streaming_{timestamp}.json"
        
        self.streaming_json_path = self.output_dir / filename
        
        # Create initial JSON structure
        initial_data = {
            "experiment": "experiment2",
            "start_time": self.experiment_start_time.isoformat() if self.experiment_start_time else datetime.now().isoformat(),
            "total_trials": 0,  # Will be updated as trials complete
            "trials": []
        }
        
        # Write initial structure
        with open(self.streaming_json_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(self.streaming_json_path)
    
    def stream_conversation_update(
        self,
        trial_number: int,
        role: str,
        content: str,
        utterance_num: int
    ) -> None:
        """
        Stream a single conversation message update to the JSON file
        
        Args:
            trial_number: Trial number
            role: "buyer" or "seller"
            content: Message content
            utterance_num: Utterance number in this trial
        """
        if self.streaming_json_path is None:
            return
        
        try:
            # Read current JSON file
            with open(self.streaming_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find or create the trial
            trial_log = None
            for trial in data.get("trials", []):
                if trial.get("trial_number") == trial_number:
                    trial_log = trial
                    break
            
            if trial_log is None:
                # Create new trial entry
                trial_log = {
                    "trial_number": trial_number,
                    "experiment_id": "experiment2",
                    "experiment_type": "single_buyer",
                    "public_conversation": [],
                    "status": "in_progress"
                }
                if "trials" not in data:
                    data["trials"] = []
                data["trials"].append(trial_log)
            
            # Add conversation message
            if "public_conversation" not in trial_log:
                trial_log["public_conversation"] = []
            
            trial_log["public_conversation"].append({
                "role": role,
                "content": content,
                "utterance_number": utterance_num,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update total trials count
            data["total_trials"] = len(data["trials"])
            data["last_updated"] = datetime.now().isoformat()
            
            # Write updated JSON back to file
            with open(self.streaming_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            # Don't crash if streaming fails, but print warning
            import traceback
            print(f"Warning: Failed to stream conversation update: {e}")
            traceback.print_exc()
    
    def stream_trial_result(self, trial_result: Dict[str, Any]) -> None:
        """
        Stream a single trial result to the JSON file
        
        Args:
            trial_result: Single trial result dictionary
        """
        if self.streaming_json_path is None:
            # Auto-start if not started
            self.start_streaming_experiment2_json()
        
        # Read current JSON file
        with open(self.streaming_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trial_number = trial_result.get("trial_number", 0)
        
        # Find existing trial or create new one
        trial_log = None
        trial_index = None
        for i, trial in enumerate(data.get("trials", [])):
            if trial.get("trial_number") == trial_number:
                trial_log = trial
                trial_index = i
                break
        
        # Preserve existing conversation if trial was already being streamed
        existing_conversation = []
        if trial_log and "public_conversation" in trial_log:
            existing_conversation = trial_log.get("public_conversation", [])
        
        # Format trial log (same structure as _write_detailed_json_logs)
        # If trial doesn't exist, create new one; otherwise update existing one
        if trial_log is None:
            trial_log = {
                "trial_number": trial_result.get("trial_number", 0),
            "experiment_id": trial_result.get("experiment_id", "experiment2"),
            "experiment_type": trial_result.get("experiment_type", "single_buyer"),
            
            # Agent information
            "agents": {
                "buyer": {
                    "name": trial_result.get("buyer_name", ""),
                    "race": trial_result.get("buyer_race", ""),
                    "gender": trial_result.get("buyer_gender", ""),
                    "budget": trial_result.get("buyer_budget", 0)
                },
                "seller": {
                    "name": trial_result.get("seller_name", ""),
                    "race": trial_result.get("seller_race", ""),
                    "gender": trial_result.get("seller_gender", ""),
                    "minimum_price": trial_result.get("seller_budget", 0)
                }
            },
            
            # House information
            "house": {
                "address": trial_result.get("house_address", "")
            },
            
            # Negotiation outcomes
            "outcomes": {
                "agreed": trial_result.get("agreed", False),
                "agreed_price": trial_result.get("agreed_price"),
                "final_price": trial_result.get("final_price"),
                "num_proposals": trial_result.get("num_proposals", 0),
                "num_utterances": trial_result.get("num_utterances", 0),
                "buyer_proposals_count": trial_result.get("buyer_proposals_count", 0),
                "seller_proposals_count": trial_result.get("seller_proposals_count", 0),
                "proposals": trial_result.get("proposals", [])
            },
            
            # PUBLIC CONVERSATION (what agents said to each other)
            # Preserve existing streamed conversation, or use from trial_result
            "public_conversation": existing_conversation if existing_conversation else trial_result.get("public_conversation", trial_result.get("conversation_history", [])),
            "status": "completed" if trial_result.get("agreed", False) else "no_agreement",
            
            # PRIVATE THOUGHTS (internal reasoning, not shared)
            "private_thoughts": trial_result.get("private_thoughts", {
                "buyer": trial_result.get("buyer_thoughts", []),
                "seller": trial_result.get("seller_thoughts", [])
            }),
            
            # DETAILED LLM INTERACTIONS (all prompts and responses)
            "llm_interactions": trial_result.get("llm_interactions", {}),
            
            # OFFERS EXTRACTED FROM TAGS (deterministic price extraction)
            "offers_from_tags": trial_result.get("offers_from_tags", []),
            
            # Breakdown by privacy level
            "interactions_by_privacy": {
                "public": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("privacy") == "public"
                ],
                "private": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("privacy") == "private"
                ]
            },
            
            # Breakdown by interaction type
            "interactions_by_type": {
                "think": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("interaction_type") == "think"
                ],
                "reflect": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("interaction_type") == "reflect"
                ],
                "discuss": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("interaction_type") == "discuss"
                ],
                "propose_price": [
                    interaction for interaction in 
                    trial_result.get("llm_interactions", {}).get("all_interactions", [])
                    if interaction.get("interaction_type") == "propose_price"
                ]
            }
        }
        
        # Update or append trial
        if trial_index is not None:
            # Update existing trial (merge with streamed data)
            data["trials"][trial_index] = trial_log
        else:
            # Append new trial
            data["trials"].append(trial_log)
        
        data["total_trials"] = len(data["trials"])
        data["last_updated"] = datetime.now().isoformat()
        
        # Write updated JSON back to file
        with open(self.streaming_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def finish_streaming_experiment2_json(self) -> str:
        """
        Finalize streaming JSON file - add end time
        
        Returns:
            Path to JSON file
        """
        if self.streaming_json_path is None:
            return None
        
        # Read current JSON file
        with open(self.streaming_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add end time
        data["end_time"] = datetime.now().isoformat()
        if self.experiment_start_time:
            duration = (datetime.now() - self.experiment_start_time).total_seconds()
            data["duration_seconds"] = duration
        
        # Write final version
        with open(self.streaming_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(self.streaming_json_path)
    
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

