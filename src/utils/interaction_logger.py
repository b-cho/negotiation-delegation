"""Logger for prompt-response interactions"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class InteractionLogger:
    """Logger for LLM prompt-response interactions"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize interaction logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.interaction_count = 0
    
    def start_session(self, experiment_id: str, trial_id: Optional[int] = None) -> str:
        """
        Start a new logging session
        
        Args:
            experiment_id: ID of the experiment
            trial_id: Optional trial number
        
        Returns:
            Session ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if trial_id is not None:
            self.current_session_id = f"{experiment_id}_trial{trial_id}_{timestamp}"
        else:
            self.current_session_id = f"{experiment_id}_{timestamp}"
        self.interaction_count = 0
        return self.current_session_id
    
    def log_interaction(
        self,
        prompt: str,
        response: str,
        interaction_type: str = "generate",
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Path:
        """
        Log a prompt-response interaction
        
        Args:
            prompt: The prompt sent to the LLM
            response: The response from the LLM
            interaction_type: Type of interaction (e.g., "generate", "think", "discuss", "propose")
            metadata: Optional metadata about the interaction
            system_prompt: Optional system prompt used
        
        Returns:
            Path to the log file
        """
        if not self.current_session_id:
            # Auto-start session if not started
            self.start_session("unknown")
        
        self.interaction_count += 1
        
        log_entry = {
            "interaction_number": self.interaction_count,
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "prompt": prompt,
            "response": response,
        }
        
        if system_prompt:
            log_entry["system_prompt"] = system_prompt
        
        if metadata:
            log_entry["metadata"] = metadata
        
        # Write to JSON file (append mode)
        log_file = self.log_dir / f"{self.current_session_id}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        return log_file
    
    def log_experiment1_trial(
        self,
        trial_id: int,
        recommendation_type: str,
        profile_name: str,
        prompt: str,
        response: str,
        recommended_price: float
    ) -> Path:
        """
        Log an Experiment 1 trial interaction
        
        Args:
            trial_id: Trial number
            recommendation_type: "buyer" or "seller"
            profile_name: Name of the buyer/seller
            prompt: Prompt sent to LLM
            response: Response from LLM
            recommended_price: Extracted recommended price
        
        Returns:
            Path to log file
        """
        session_id = self.start_session("experiment1", trial_id)
        
        metadata = {
            "recommendation_type": recommendation_type,
            "profile_name": profile_name,
            "recommended_price": recommended_price
        }
        
        return self.log_interaction(
            prompt=prompt,
            response=response,
            interaction_type=f"recommendation_{recommendation_type}",
            metadata=metadata
        )
    
    def log_agent_interaction(
        self,
        agent_role: str,
        agent_name: str,
        interaction_type: str,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Log an agent interaction (for Experiment 2)
        
        Args:
            agent_role: Role of the agent ("buyer" or "seller")
            agent_name: Name of the agent
            interaction_type: Type of interaction ("think", "reflect", "discuss", "propose")
            prompt: Prompt sent to LLM
            response: Response from LLM
            system_prompt: Optional system prompt
            metadata: Optional metadata
        
        Returns:
            Path to log file
        """
        if not self.current_session_id:
            self.start_session("experiment2")
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "agent_role": agent_role,
            "agent_name": agent_name
        })
        
        return self.log_interaction(
            prompt=prompt,
            response=response,
            interaction_type=f"{agent_role}_{interaction_type}",
            system_prompt=system_prompt,
            metadata=metadata
        )

