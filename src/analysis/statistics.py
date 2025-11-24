"""Statistical analysis for experiment results"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


class StatisticalAnalyzer:
    """Statistical analysis for experiment results"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer
        
        Args:
            significance_level: Significance level for tests (default 0.05)
        """
        self.significance_level = significance_level
    
    def analyze_experiment1(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Experiment 1 results (price recommendations)
        
        Args:
            results: List of trial results from Experiment 1
        
        Returns:
            Dictionary with statistical analysis results
        """
        df = pd.DataFrame(results)
        
        analysis = {
            "experiment": "experiment1",
            "total_trials": len(results),
            "tests": []
        }
        
        # Group by buyer race
        race_groups = df.groupby("buyer_race")["recommended_price"].apply(list).to_dict()
        if len(race_groups) > 1:
            race_test = self._run_t_test_groups(race_groups, "buyer_race")
            analysis["tests"].append(race_test)
            
            if len(race_groups) > 2:
                race_anova = self._run_anova(race_groups, "buyer_race")
                analysis["tests"].append(race_anova)
        
        # Group by buyer gender
        gender_groups = df.groupby("buyer_gender")["recommended_price"].apply(list).to_dict()
        if len(gender_groups) > 1:
            gender_test = self._run_t_test_groups(gender_groups, "buyer_gender")
            analysis["tests"].append(gender_test)
        
        # Group by seller race
        seller_race_groups = df.groupby("seller_race")["recommended_price"].apply(list).to_dict()
        if len(seller_race_groups) > 1:
            seller_race_test = self._run_t_test_groups(seller_race_groups, "seller_race")
            analysis["tests"].append(seller_race_test)
        
        # Group by seller gender
        seller_gender_groups = df.groupby("seller_gender")["recommended_price"].apply(list).to_dict()
        if len(seller_gender_groups) > 1:
            seller_gender_test = self._run_t_test_groups(seller_gender_groups, "seller_gender")
            analysis["tests"].append(seller_gender_test)
        
        # Group by race and gender combination
        df["buyer_demographic"] = df["buyer_race"] + "_" + df["buyer_gender"]
        demo_groups = df.groupby("buyer_demographic")["recommended_price"].apply(list).to_dict()
        if len(demo_groups) > 2:
            demo_anova = self._run_anova(demo_groups, "buyer_race_and_gender")
            analysis["tests"].append(demo_anova)
        
        # Summary statistics
        analysis["summary"] = {
            "mean_price": df["recommended_price"].mean(),
            "std_price": df["recommended_price"].std(),
            "min_price": df["recommended_price"].min(),
            "max_price": df["recommended_price"].max(),
            "by_buyer_race": df.groupby("buyer_race")["recommended_price"].agg(["mean", "std", "count"]).to_dict("index"),
            "by_buyer_gender": df.groupby("buyer_gender")["recommended_price"].agg(["mean", "std", "count"]).to_dict("index"),
        }
        
        return analysis
    
    def analyze_experiment2(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze Experiment 2 results (negotiation outcomes)
        
        Args:
            results: List of trial results from Experiment 2
        
        Returns:
            Dictionary with statistical analysis results
        """
        df = pd.DataFrame(results)
        
        analysis = {
            "experiment": "experiment2",
            "total_trials": len(results),
            "tests": []
        }
        
        # Filter to agreed negotiations only
        agreed_df = df[df["agreed"] == True].copy()
        
        if len(agreed_df) > 0:
            # Group by buyer race (agreed price)
            race_groups = agreed_df.groupby("buyer_race")["agreed_price"].apply(list).to_dict()
            if len(race_groups) > 1:
                race_test = self._run_t_test_groups(race_groups, "buyer_race", metric="agreed_price")
                analysis["tests"].append(race_test)
                
                if len(race_groups) > 2:
                    race_anova = self._run_anova(race_groups, "buyer_race", metric="agreed_price")
                    analysis["tests"].append(race_anova)
            
            # Group by buyer gender (agreed price)
            gender_groups = agreed_df.groupby("buyer_gender")["agreed_price"].apply(list).to_dict()
            if len(gender_groups) > 1:
                gender_test = self._run_t_test_groups(gender_groups, "buyer_gender", metric="agreed_price")
                analysis["tests"].append(gender_test)
            
            # Group by race and gender combination
            agreed_df["buyer_demographic"] = agreed_df["buyer_race"] + "_" + agreed_df["buyer_gender"]
            demo_groups = agreed_df.groupby("buyer_demographic")["agreed_price"].apply(list).to_dict()
            if len(demo_groups) > 2:
                demo_anova = self._run_anova(demo_groups, "buyer_race_and_gender", metric="agreed_price")
                analysis["tests"].append(demo_anova)
        
        # Agreement rate analysis
        agreement_rate_by_race = df.groupby("buyer_race")["agreed"].mean().to_dict()
        agreement_rate_by_gender = df.groupby("buyer_gender")["agreed"].mean().to_dict()
        
        # Number of proposals analysis
        proposal_groups = df.groupby("buyer_race")["num_proposals"].apply(list).to_dict()
        if len(proposal_groups) > 1:
            proposal_test = self._run_t_test_groups(proposal_groups, "buyer_race", metric="num_proposals")
            analysis["tests"].append(proposal_test)
        
        # Summary statistics
        analysis["summary"] = {
            "total_agreements": df["agreed"].sum(),
            "agreement_rate": df["agreed"].mean(),
            "agreement_rate_by_race": agreement_rate_by_race,
            "agreement_rate_by_gender": agreement_rate_by_gender,
            "mean_proposals": df["num_proposals"].mean(),
            "mean_agreed_price": agreed_df["agreed_price"].mean() if len(agreed_df) > 0 else None,
            "by_buyer_race": {
                "agreement_rate": agreement_rate_by_race,
                "mean_agreed_price": agreed_df.groupby("buyer_race")["agreed_price"].mean().to_dict() if len(agreed_df) > 0 else {},
                "mean_proposals": df.groupby("buyer_race")["num_proposals"].mean().to_dict()
            }
        }
        
        return analysis
    
    def _run_t_test_groups(
        self,
        groups: Dict[str, List[float]],
        group_name: str,
        metric: str = "price"
    ) -> Dict[str, Any]:
        """
        Run t-tests between all pairs of groups
        
        Args:
            groups: Dictionary mapping group names to lists of values
            group_name: Name of the grouping variable
            metric: Name of the metric being tested
        
        Returns:
            Dictionary with test results
        """
        group_names = list(groups.keys())
        comparisons = []
        
        for group1, group2 in combinations(group_names, 2):
            data1 = np.array(groups[group1])
            data2 = np.array(groups[group2])
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
            
            comparisons.append({
                "group1": group1,
                "group2": group2,
                "group1_mean": float(np.mean(data1)),
                "group2_mean": float(np.mean(data2)),
                "group1_std": float(np.std(data1)),
                "group2_std": float(np.std(data2)),
                "group1_n": len(data1),
                "group2_n": len(data2),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.significance_level,
                "cohens_d": float(cohens_d),
                "effect_size": self._interpret_effect_size(abs(cohens_d))
            })
        
        return {
            "test_type": "t_test",
            "grouping_variable": group_name,
            "metric": metric,
            "significance_level": self.significance_level,
            "comparisons": comparisons
        }
    
    def _run_anova(
        self,
        groups: Dict[str, List[float]],
        group_name: str,
        metric: str = "price"
    ) -> Dict[str, Any]:
        """
        Run ANOVA test across all groups
        
        Args:
            groups: Dictionary mapping group names to lists of values
            group_name: Name of the grouping variable
            metric: Name of the metric being tested
        
        Returns:
            Dictionary with ANOVA results
        """
        group_data = [groups[name] for name in groups.keys()]
        group_names = list(groups.keys())
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate effect size (eta squared)
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in group_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        group_stats = {
            name: {
                "mean": float(np.mean(groups[name])),
                "std": float(np.std(groups[name])),
                "n": len(groups[name])
            }
            for name in group_names
        }
        
        return {
            "test_type": "anova",
            "grouping_variable": group_name,
            "metric": metric,
            "significance_level": self.significance_level,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < self.significance_level,
            "eta_squared": float(eta_squared),
            "effect_size": self._interpret_effect_size(np.sqrt(eta_squared)),
            "group_statistics": group_stats
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size"""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

