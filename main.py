"""Main entry point for running experiments"""
import argparse
import sys
from pathlib import Path

from src.models.llm_client import LLMClient
from src.data.profiles import load_profiles_from_config
from src.data.house_specs import load_house_specs_from_config
from src.experiments.experiment1_mvp import Experiment1MVP
from src.experiments.experiment2_negotiation import Experiment2Negotiation
from src.analysis.statistics import StatisticalAnalyzer
from src.utils.config_loader import load_config, get_model_config, get_statistical_config
from src.utils.results_writer import ResultsWriter
from src.utils.interaction_logger import InteractionLogger


def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run real estate bias assessment experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="Which experiment to run (1, 2, or both)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run statistical analysis on results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize LLM client
    print("Initializing LLM client...")
    model_config = get_model_config(config)
    llm_client = LLMClient.from_config(model_config)
    print(f"Using model: {model_config['provider']} - {model_config['model']}")
    
    # Load house specifications
    print("Loading house specifications...")
    house_specs = load_house_specs_from_config(config)
    
    # Load profiles
    print("Loading buyer and seller profiles...")
    buyer_profiles, seller_profiles = load_profiles_from_config(config)
    print(f"Loaded {len(buyer_profiles)} buyer profiles and {len(seller_profiles)} seller profiles")
    
    # Initialize results writer and logger
    results_writer = ResultsWriter(output_dir=args.output_dir)
    logger = InteractionLogger(log_dir="logs")
    
    # Run experiments
    experiment1_results = None
    experiment2_results = None
    
    if args.experiment in ["1", "both"]:
        print("\n" + "="*60)
        print("Running Experiment 1: MVP Price Recommendation")
        print("="*60)
        
        experiment1 = Experiment1MVP(
            llm_client=llm_client,
            house_specs=house_specs,
            config=config,
            logger=logger
        )
        
        experiment1_results = experiment1.run_experiment(
            buyer_profiles=buyer_profiles,
            seller_profiles=seller_profiles
        )
        
        # Write results
        csv_path = results_writer.write_experiment1_results(experiment1_results)
        print(f"\nExperiment 1 results written to: {csv_path}")
        print(f"Total trials: {len(experiment1_results)}")
    
    if args.experiment in ["2", "both"]:
        print("\n" + "="*60)
        print("Running Experiment 2: Full Negotiation")
        print("="*60)
        
        experiment2 = Experiment2Negotiation(
            llm_client=llm_client,
            house_specs=house_specs,
            config=config
        )
        
        experiment2_results = experiment2.run_experiment(
            buyer_profiles=buyer_profiles,
            seller_profiles=seller_profiles
        )
        
        # Write results
        csv_path = results_writer.write_experiment2_results(experiment2_results)
        print(f"\nExperiment 2 results written to: {csv_path}")
        print(f"Total trials: {len(experiment2_results)}")
        
        # Print summary
        agreed_count = sum(1 for r in experiment2_results if r.get("agreed", False))
        print(f"Agreements: {agreed_count}/{len(experiment2_results)} ({100*agreed_count/len(experiment2_results):.1f}%)")
    
    # Run statistical analysis if requested
    if args.analyze:
        print("\n" + "="*60)
        print("Running Statistical Analysis")
        print("="*60)
        
        stats_config = get_statistical_config(config)
        analyzer = StatisticalAnalyzer(
            significance_level=stats_config.get("significance_level", 0.05)
        )
        
        if experiment1_results:
            print("\nAnalyzing Experiment 1 results...")
            analysis1 = analyzer.analyze_experiment1(experiment1_results)
            json_path = results_writer.write_statistical_analysis(analysis1)
            print(f"Analysis written to: {json_path}")
            
            # Print key findings
            print("\nKey Findings (Experiment 1):")
            for test in analysis1.get("tests", []):
                if test.get("test_type") == "t_test":
                    print(f"\n{test['grouping_variable']} - t-tests:")
                    for comp in test.get("comparisons", []):
                        sig = "***" if comp["significant"] else ""
                        print(f"  {comp['group1']} vs {comp['group2']}: "
                              f"mean1=${comp['group1_mean']:,.0f}, "
                              f"mean2=${comp['group2_mean']:,.0f}, "
                              f"p={comp['p_value']:.4f} {sig}")
                elif test.get("test_type") == "anova":
                    sig = "***" if test["significant"] else ""
                    print(f"\n{test['grouping_variable']} - ANOVA: "
                          f"F={test['f_statistic']:.2f}, "
                          f"p={test['p_value']:.4f} {sig}")
        
        if experiment2_results:
            print("\nAnalyzing Experiment 2 results...")
            analysis2 = analyzer.analyze_experiment2(experiment2_results)
            json_path = results_writer.write_statistical_analysis(analysis2)
            print(f"Analysis written to: {json_path}")
            
            # Print key findings
            print("\nKey Findings (Experiment 2):")
            summary = analysis2.get("summary", {})
            print(f"Overall agreement rate: {summary.get('agreement_rate', 0):.2%}")
            print(f"\nAgreement rate by buyer race:")
            for race, rate in summary.get("agreement_rate_by_race", {}).items():
                print(f"  {race}: {rate:.2%}")
            
            for test in analysis2.get("tests", []):
                if test.get("test_type") == "t_test":
                    print(f"\n{test['grouping_variable']} - t-tests ({test['metric']}):")
                    for comp in test.get("comparisons", []):
                        sig = "***" if comp["significant"] else ""
                        print(f"  {comp['group1']} vs {comp['group2']}: "
                              f"mean1=${comp['group1_mean']:,.0f}, "
                              f"mean2=${comp['group2_mean']:,.0f}, "
                              f"p={comp['p_value']:.4f} {sig}")
    
    print("\n" + "="*60)
    print("Experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()

