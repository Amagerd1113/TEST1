#!/usr/bin/env python3
"""
Run comprehensive evaluation suite for VLA-GR navigation.
Generates real performance metrics through systematic testing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import VLAGREvaluator, EvaluationMetrics


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluation:
    """Run complete evaluation suite and generate metrics report."""
    
    def __init__(self, args):
        self.args = args
        self.results_dir = Path(args.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = VLAGREvaluator(
            model_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            output_dir=args.output_dir
        )
        
        self.all_results = {}
        
    def run(self):
        """Run complete evaluation pipeline."""
        
        logger.info("="*60)
        logger.info("Starting VLA-GR Comprehensive Evaluation")
        logger.info("="*60)
        
        # 1. Standard benchmarks
        if not self.args.skip_standard:
            logger.info("\n1. Running Standard Benchmarks...")
            standard_results = self.evaluator.evaluate_standard_benchmarks(
                num_episodes=self.args.num_episodes,
                splits=["val", "test"] if not self.args.val_only else ["val"]
            )
            self.all_results["standard"] = standard_results
            self._print_standard_results(standard_results)
        
        # 2. Ablation studies
        if not self.args.skip_ablation:
            logger.info("\n2. Running Ablation Studies...")
            ablation_results = self.evaluator.run_ablation_studies()
            self.all_results["ablation"] = ablation_results
            self._print_ablation_results(ablation_results)
        
        # 3. Baseline comparison
        if not self.args.skip_baselines:
            logger.info("\n3. Comparing with Baselines...")
            baseline_results = self.evaluator.compare_with_baselines(
                num_episodes=min(self.args.num_episodes, 100)
            )
            self.all_results["baselines"] = baseline_results.to_dict()
            print("\nBaseline Comparison:")
            print(baseline_results)
        
        # 4. Robustness testing
        if not self.args.skip_robustness:
            logger.info("\n4. Testing Robustness...")
            robustness_results = self.evaluator.test_robustness()
            self.all_results["robustness"] = robustness_results
            self._print_robustness_results(robustness_results)
        
        # 5. Generate comprehensive report
        self.generate_report()
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Complete!")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*60)
        
    def _print_standard_results(self, results):
        """Print standard benchmark results."""
        
        print("\nStandard Benchmark Results:")
        print("-" * 40)
        
        for split, metrics in results.items():
            print(f"\n{split.upper()} Split:")
            print(f"  Success Rate: {metrics.success_rate:.2%}")
            print(f"  SPL: {metrics.spl:.3f}")
            print(f"  Navigation Error: {metrics.navigation_error:.2f}m")
            print(f"  Collision Rate: {metrics.collision_rate:.2%}")
            print(f"  Avg Steps: {metrics.episode_length:.1f}")
            print(f"  Inference Time: {metrics.inference_time*1000:.1f}ms")
            
    def _print_ablation_results(self, results):
        """Print ablation study results."""
        
        print("\nAblation Study Results:")
        print("-" * 40)
        
        full_model = results.get("full_model")
        if not full_model:
            return
            
        print(f"\nFull Model Baseline:")
        print(f"  Success Rate: {full_model.success_rate:.2%}")
        print(f"  SPL: {full_model.spl:.3f}")
        
        print("\nAblation Impact (relative drop):")
        
        for name, metrics in results.items():
            if name == "full_model":
                continue
                
            success_drop = (full_model.success_rate - metrics.success_rate) / full_model.success_rate
            spl_drop = (full_model.spl - metrics.spl) / full_model.spl
            
            print(f"\n  {name}:")
            print(f"    Success Rate: -{success_drop:.1%}")
            print(f"    SPL: -{spl_drop:.1%}")
            
    def _print_robustness_results(self, results):
        """Print robustness test results."""
        
        print("\nRobustness Test Results:")
        print("-" * 40)
        
        if "occlusion" in results:
            print("\nOcclusion Robustness:")
            for item in results["occlusion"]:
                print(f"  {item['occlusion']*100:.0f}% occlusion: "
                      f"Success={item['success_rate']:.2%}")
                      
        if "noise" in results:
            print("\nNoise Robustness:")
            for item in results["noise"]:
                print(f"  Noise={item['noise']:.2f}: "
                      f"Success={item['success_rate']:.2%}")
                      
    def generate_report(self):
        """Generate comprehensive HTML report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VLA-GR Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #0066cc; }}
                .timestamp {{ color: #999; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>VLA-GR Navigation Evaluation Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            {self._generate_standard_section()}
            {self._generate_ablation_section()}
            {self._generate_baseline_section()}
            {self._generate_robustness_section()}
            
            <h2>Configuration</h2>
            <pre>{json.dumps(self.args.__dict__, indent=2)}</pre>
        </body>
        </html>
        """
        
        report_path = self.results_dir / "evaluation_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to: {report_path}")
        
    def _generate_standard_section(self):
        """Generate HTML section for standard results."""
        
        if "standard" not in self.all_results:
            return ""
            
        html = "<h2>Standard Benchmark Results</h2>"
        html += "<table>"
        html += "<tr><th>Metric</th><th>Val</th><th>Test</th></tr>"
        
        metrics_to_show = [
            ("Success Rate", "success_rate", True),
            ("SPL", "spl", False),
            ("Navigation Error (m)", "navigation_error", False),
            ("Collision Rate", "collision_rate", True),
            ("Avg Episode Length", "episode_length", False),
            ("Inference Time (ms)", "inference_time", False)
        ]
        
        for metric_name, metric_key, is_percentage in metrics_to_show:
            html += f"<tr><td>{metric_name}</td>"
            
            for split in ["val", "test"]:
                if split in self.all_results["standard"]:
                    value = getattr(self.all_results["standard"][split], metric_key)
                    
                    if metric_key == "inference_time":
                        value = value * 1000  # Convert to ms
                        
                    if is_percentage:
                        html += f"<td class='metric'>{value:.2%}</td>"
                    else:
                        html += f"<td class='metric'>{value:.2f}</td>"
                else:
                    html += "<td>-</td>"
                    
            html += "</tr>"
            
        html += "</table>"
        return html
        
    def _generate_ablation_section(self):
        """Generate HTML section for ablation results."""
        
        if "ablation" not in self.all_results:
            return ""
            
        html = "<h2>Ablation Studies</h2>"
        html += "<table>"
        html += "<tr><th>Model Variant</th><th>Success Rate</th><th>SPL</th><th>Impact</th></tr>"
        
        full_model = self.all_results["ablation"].get("full_model")
        
        for name, metrics in self.all_results["ablation"].items():
            if full_model and name != "full_model":
                impact = (full_model.success_rate - metrics.success_rate) / full_model.success_rate
                impact_str = f"-{impact:.1%}"
            else:
                impact_str = "-"
                
            html += f"<tr>"
            html += f"<td>{name.replace('_', ' ').title()}</td>"
            html += f"<td class='metric'>{metrics.success_rate:.2%}</td>"
            html += f"<td class='metric'>{metrics.spl:.3f}</td>"
            html += f"<td>{impact_str}</td>"
            html += "</tr>"
            
        html += "</table>"
        return html
        
    def _generate_baseline_section(self):
        """Generate HTML section for baseline comparison."""
        
        if "baselines" not in self.all_results:
            return ""
            
        html = "<h2>Baseline Comparison</h2>"
        
        # Convert dict back to DataFrame for display
        df = pd.DataFrame(self.all_results["baselines"])
        html += df.to_html(classes='metric', index=False)
        
        return html
        
    def _generate_robustness_section(self):
        """Generate HTML section for robustness results."""
        
        if "robustness" not in self.all_results:
            return ""
            
        html = "<h2>Robustness Testing</h2>"
        
        if "occlusion" in self.all_results["robustness"]:
            html += "<h3>Occlusion Robustness</h3>"
            html += "<table>"
            html += "<tr><th>Occlusion Level</th><th>Success Rate</th></tr>"
            
            for item in self.all_results["robustness"]["occlusion"]:
                html += f"<tr>"
                html += f"<td>{item['occlusion']*100:.0f}%</td>"
                html += f"<td class='metric'>{item['success_rate']:.2%}</td>"
                html += "</tr>"
                
            html += "</table>"
            
        return html


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Run VLA-GR Evaluation Suite")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device"
    )
    
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="Only evaluate on validation set"
    )
    
    parser.add_argument(
        "--skip_standard",
        action="store_true",
        help="Skip standard benchmarks"
    )
    
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="Skip ablation studies"
    )
    
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline comparison"
    )
    
    parser.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip robustness testing"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluation = ComprehensiveEvaluation(args)
    evaluation.run()


if __name__ == "__main__":
    main()
