#!/usr/bin/env python3
"""
CAR T-Cell RL Training Metrics Analyzer

This script analyzes the learning trends of the PPO agent trained on the synthetic
CAR T-cell manufacturing environment by examining key training metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.signal import savgol_filter
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class RLTrainingAnalyzer:
    def __init__(self, tensorboard_dir):
        """Initialize the analyzer with the tensorboard directory path."""
        self.tensorboard_dir = Path(tensorboard_dir)
        self.metrics_data = {}
        self.analysis_results = {}
        
        # Define metric mappings and their expected behaviors
        self.metric_info = {
            'train_explained_variance': {
                'file_pattern': '*explained_variance.csv',
                'expected_trend': 'increasing',
                'ideal_range': (0.8, 1.0),
                'description': 'How well the value function predicts returns'
            },
            'train_loss': {
                'file_pattern': '*train_loss.csv',
                'expected_trend': 'decreasing',
                'ideal_range': (None, None),
                'description': 'Overall training loss (should decrease)'
            },
            'policy_gradient_loss': {
                'file_pattern': '*policy_gradient_loss.csv',
                'expected_trend': 'decreasing',
                'ideal_range': (None, None),
                'description': 'Policy gradient loss (should decrease)'
            },
            'value_loss': {
                'file_pattern': '*value_loss.csv',
                'expected_trend': 'decreasing',
                'ideal_range': (None, None),
                'description': 'Value function loss (should decrease)'
            },
            'entropy_loss': {
                'file_pattern': '*entropy_loss.csv',
                'expected_trend': 'stabilizing',
                'ideal_range': (None, None),
                'description': 'Entropy loss (exploration vs exploitation balance)'
            },
            'ep_rew_mean': {
                'file_pattern': '*ep_rew_mean.csv',
                'expected_trend': 'increasing',
                'ideal_range': (None, None),
                'description': 'Mean episode reward (should increase as agent learns)'
            }
        }
    
    def load_metrics_data(self):
        """Load all CSV files containing training metrics."""
        print("🔍 Loading training metrics data...")
        
        for metric_name, info in self.metric_info.items():
            csv_files = list(self.tensorboard_dir.rglob(info['file_pattern']))
            
            if csv_files:
                # Take the first matching file
                csv_file = csv_files[0]
                print(f"   📊 Loading {metric_name} from {csv_file.name}")
                
                try:
                    df = pd.read_csv(csv_file)
                    # Ensure we have the expected columns
                    if 'Step' in df.columns and 'Value' in df.columns:
                        self.metrics_data[metric_name] = df
                    else:
                        print(f"   ⚠️  Warning: {csv_file.name} missing expected columns")
                except Exception as e:
                    print(f"   ❌ Error loading {csv_file.name}: {e}")
            else:
                print(f"   ⚠️  No file found for {metric_name}")
        
        print(f"✅ Loaded {len(self.metrics_data)} metrics successfully\n")
    
    def calculate_trend_statistics(self, values, steps):
        """Calculate trend statistics for a metric."""
        # Remove any NaN or infinite values
        mask = np.isfinite(values)
        clean_values = values[mask]
        clean_steps = steps[mask]
        
        if len(clean_values) < 10:
            return None
        
        # Calculate linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_steps, clean_values)
        
        # Calculate smoothed trend using Savitzky-Golay filter
        if len(clean_values) > 50:
            window_length = min(51, len(clean_values) // 3)
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_values = savgol_filter(clean_values, window_length, 3)
        else:
            smoothed_values = clean_values
        
        # Calculate convergence metrics
        recent_portion = int(0.2 * len(clean_values))  # Last 20% of training
        if recent_portion > 10:
            recent_values = clean_values[-recent_portion:]
            convergence_std = np.std(recent_values)
            convergence_mean = np.mean(recent_values)
        else:
            convergence_std = np.std(clean_values)
            convergence_mean = np.mean(clean_values)
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'initial_value': clean_values[0],
            'final_value': clean_values[-1],
            'min_value': np.min(clean_values),
            'max_value': np.max(clean_values),
            'convergence_std': convergence_std,
            'convergence_mean': convergence_mean,
            'smoothed_values': smoothed_values,
            'clean_steps': clean_steps,
            'clean_values': clean_values
        }
    
    def analyze_learning_patterns(self):
        """Analyze learning patterns for each metric."""
        print("🧠 Analyzing learning patterns...")
        
        for metric_name, df in self.metrics_data.items():
            print(f"   📈 Analyzing {metric_name}...")
            
            steps = df['Step'].values
            values = df['Value'].values
            
            stats = self.calculate_trend_statistics(values, steps)
            if stats is None:
                print(f"   ⚠️  Insufficient data for {metric_name}")
                continue
            
            # Determine if the trend matches expectations
            expected_trend = self.metric_info[metric_name]['expected_trend']
            trend_analysis = self.interpret_trend(stats, expected_trend)
            
            self.analysis_results[metric_name] = {
                'statistics': stats,
                'trend_analysis': trend_analysis,
                'metric_info': self.metric_info[metric_name]
            }
    
    def interpret_trend(self, stats, expected_trend):
        """Interpret whether the trend matches expectations."""
        slope = stats['slope']
        r_squared = stats['r_squared']
        
        # Determine actual trend direction
        if abs(slope) < 1e-10:
            actual_trend = 'stable'
        elif slope > 0:
            actual_trend = 'increasing'
        else:
            actual_trend = 'decreasing'
        
        # Check if trend strength is significant
        trend_strength = 'weak' if r_squared < 0.3 else 'moderate' if r_squared < 0.7 else 'strong'
        
        # Determine if trend matches expectations
        if expected_trend == 'stabilizing':
            # For stabilizing metrics, we want low variance in recent values
            is_good = stats['convergence_std'] < abs(stats['convergence_mean']) * 0.1
            status = 'converging' if is_good else 'unstable'
        else:
            is_good = (expected_trend == actual_trend)
            status = 'good' if is_good else 'concerning'
        
        return {
            'actual_trend': actual_trend,
            'expected_trend': expected_trend,
            'trend_strength': trend_strength,
            'r_squared': r_squared,
            'status': status,
            'is_learning_well': is_good
        }
    
    def generate_learning_summary(self):
        """Generate a comprehensive learning summary."""
        print("\n" + "="*60)
        print("🎯 CAR T-CELL RL AGENT LEARNING ANALYSIS SUMMARY")
        print("="*60)
        
        total_metrics = len(self.analysis_results)
        learning_well = sum(1 for result in self.analysis_results.values() 
                           if result['trend_analysis']['is_learning_well'])
        
        print(f"\n📊 OVERALL LEARNING STATUS: {learning_well}/{total_metrics} metrics show good learning")
        
        if learning_well >= total_metrics * 0.75:
            overall_status = "🟢 EXCELLENT - Agent is learning effectively"
        elif learning_well >= total_metrics * 0.5:
            overall_status = "🟡 GOOD - Agent is learning with some concerns"
        else:
            overall_status = "🔴 CONCERNING - Agent may have learning issues"
        
        print(f"Overall Assessment: {overall_status}\n")
        
        # Detailed metric analysis
        for metric_name, result in self.analysis_results.items():
            stats = result['statistics']
            trend = result['trend_analysis']
            info = result['metric_info']
            
            print(f"📈 {metric_name.upper().replace('_', ' ')}")
            print(f"   Description: {info['description']}")
            print(f"   Trend: {trend['actual_trend']} ({trend['trend_strength']}, R²={trend['r_squared']:.3f})")
            print(f"   Status: {trend['status'].upper()}")
            print(f"   Value change: {stats['initial_value']:.4f} → {stats['final_value']:.4f}")
            
            # Special interpretations
            if metric_name == 'train_explained_variance':
                if stats['final_value'] > 0.8:
                    print("   ✅ Agent shows strong understanding of environment")
                elif stats['final_value'] > 0.5:
                    print("   ⚠️  Agent has moderate understanding")
                else:
                    print("   ❌ Agent struggles to understand environment")
            
            if metric_name == 'ep_rew_mean':
                reward_improvement = stats['final_value'] - stats['initial_value']
                if reward_improvement > 0:
                    print(f"   ✅ Reward improved by {reward_improvement:.4f}")
                    if trend['actual_trend'] == 'increasing':
                        print("   ✅ Agent is successfully learning to maximize rewards")
                else:
                    print(f"   ❌ Reward decreased by {abs(reward_improvement):.4f}")
                    print("   ⚠️  Agent may not be learning optimal policy")
            
            if 'loss' in metric_name and trend['actual_trend'] == 'decreasing':
                print("   ✅ Loss is decreasing as expected")
            elif 'loss' in metric_name:
                print("   ⚠️  Loss not decreasing - potential training issues")
            
            print()
        
        return overall_status
    
    def create_visualizations(self, save_plots=True):
        """Create individual visualizations for each training metric."""
        print("📊 Creating individual training visualizations...")

        n_metrics = len(self.analysis_results)
        if n_metrics == 0:
            print("   ❌ No metrics to visualize")
            return

        # Set larger font sizes for better readability
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20
        })

        # Create individual plots for each metric
        for idx, (metric_name, result) in enumerate(self.analysis_results.items()):
            # Create individual figure for each metric
            fig, ax = plt.subplots(figsize=(12, 8))

            stats = result['statistics']
            trend = result['trend_analysis']
            info = result['metric_info']

            # Plot original data with larger line width
            ax.plot(stats['clean_steps'], stats['clean_values'],
                   alpha=0.4, color='gray', linewidth=1.5, label='Raw data')

            # Plot smoothed trend with larger line width
            ax.plot(stats['clean_steps'], stats['smoothed_values'],
                   linewidth=3.0, color='blue', label='Smoothed trend')

            # Add trend line with larger line width
            trend_line = stats['slope'] * stats['clean_steps'] + stats['initial_value']
            ax.plot(stats['clean_steps'], trend_line, '--',
                   alpha=0.8, color='red', linewidth=2.5,
                   label=f'Linear trend (R²={trend["r_squared"]:.3f})')

            # Enhanced formatting with larger fonts
            title_text = f'{metric_name.replace("_", " ").title()}\nStatus: {trend["status"].title()}'
            ax.set_title(title_text, fontweight='bold', fontsize=20, pad=20)

            ax.set_xlabel('Training Steps', fontsize=16, fontweight='bold')
            ax.set_ylabel('Value', fontsize=16, fontweight='bold')

            # Add description as text box
            desc_text = f"Description: {info['description']}\nExpected: {info['expected_trend']}"
            ax.text(0.02, 0.98, desc_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

            # Enhanced legend
            ax.legend(fontsize=14, loc='upper right', framealpha=0.9)

            # Enhanced grid
            ax.grid(True, alpha=0.4, linewidth=1.2)

            # Color code background based on learning status
            if trend['is_learning_well']:
                ax.patch.set_facecolor('lightgreen')
                ax.patch.set_alpha(0.1)
                # Add success indicators
                ax.text(0.98, 0.02, '✓ Good Learning',
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            else:
                ax.patch.set_facecolor('lightcoral')
                ax.patch.set_alpha(0.1)
                # Add warning indicators
                ax.text(0.98, 0.02, '⚠ Concerning',
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))

            # Add statistical summary
            stats_text = '.6f'
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                   verticalalignment='bottom', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6))

            plt.tight_layout()

            if save_plots:
                # Save individual plot
                plot_filename = f'{metric_name}_analysis.png'
                plot_path = self.tensorboard_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   💾 Saved {metric_name} plot to {plot_path}")

            plt.show()
            plt.close()  # Close the figure to free memory

        # Create a summary plot showing all metrics overview
        if save_plots and n_metrics > 1:
            self.create_summary_overview()

    def create_summary_overview(self):
        """Create a summary overview plot showing all metrics in a compact format."""
        print("   📊 Creating summary overview plot...")

        n_metrics = len(self.analysis_results)
        rows = (n_metrics + 2) // 3  # Arrange in roughly 3 columns
        cols = min(3, n_metrics)

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        fig.suptitle('CAR T-Cell RL Agent Training Analysis - Overview',
                    fontsize=20, fontweight='bold', y=0.98)

        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (metric_name, result) in enumerate(self.analysis_results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            stats = result['statistics']
            trend = result['trend_analysis']

            # Simplified plotting for overview
            ax.plot(stats['clean_steps'], stats['clean_values'],
                   alpha=0.5, color='gray', linewidth=1.0, label='Raw data')
            ax.plot(stats['clean_steps'], stats['smoothed_values'],
                   linewidth=2.0, label='Trend')

            # Simplified title
            status_color = 'green' if trend['is_learning_well'] else 'red'
            ax.set_title(f'{metric_name.replace("_", " ").title()}',
                        fontweight='bold', fontsize=14, color=status_color)
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Color background
            if trend['is_learning_well']:
                ax.patch.set_facecolor('lightgreen')
                ax.patch.set_alpha(0.1)
            else:
                ax.patch.set_facecolor('lightcoral')
                ax.patch.set_alpha(0.1)

        # Hide unused subplots
        for idx in range(len(self.analysis_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save summary plot
        summary_path = self.tensorboard_dir / 'training_analysis_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Summary overview saved to {summary_path}")

        plt.show()
        plt.close()

    def export_analysis_report(self):
        """Export detailed analysis report to text file."""
        report_path = self.tensorboard_dir / 'training_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("CAR T-CELL RL AGENT TRAINING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for metric_name, result in self.analysis_results.items():
                stats = result['statistics']
                trend = result['trend_analysis']
                info = result['metric_info']
                
                f.write(f"METRIC: {metric_name.upper()}\n")
                f.write(f"Description: {info['description']}\n")
                f.write(f"Expected Trend: {info['expected_trend']}\n")
                f.write(f"Actual Trend: {trend['actual_trend']}\n")
                f.write(f"Trend Strength: {trend['trend_strength']} (R² = {trend['r_squared']:.4f})\n")
                f.write(f"Learning Status: {trend['status']}\n")
                f.write(f"Initial Value: {stats['initial_value']:.6f}\n")
                f.write(f"Final Value: {stats['final_value']:.6f}\n")
                f.write(f"Min Value: {stats['min_value']:.6f}\n")
                f.write(f"Max Value: {stats['max_value']:.6f}\n")
                f.write(f"Convergence Std: {stats['convergence_std']:.6f}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"📄 Detailed report exported to {report_path}")
    
    def run_complete_analysis(self, create_plots=True, export_report=True):
        """Run the complete analysis pipeline."""
        print("🚀 Starting complete CAR T-Cell RL training analysis...\n")
        
        # Step 1: Load data
        self.load_metrics_data()
        
        if not self.metrics_data:
            print("❌ No metrics data found. Please check your CSV files.")
            return
        
        # Step 2: Analyze patterns
        self.analyze_learning_patterns()
        
        # Step 3: Generate summary
        overall_status = self.generate_learning_summary()
        
        # Step 4: Create visualizations
        if create_plots:
            self.create_visualizations()
        
        # Step 5: Export report
        if export_report:
            self.export_analysis_report()
        
        print("\n✅ Analysis complete!")
        return overall_status


def main():
    parser = argparse.ArgumentParser(description='Analyze CAR T-Cell RL training metrics')
    parser.add_argument('--tensorboard-dir', 
                       default='results/ppo_cart_tensorboard',
                       help='Path to tensorboard directory containing CSV files')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip exporting detailed report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RLTrainingAnalyzer(args.tensorboard_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis(
        create_plots=not args.no_plots,
        export_report=not args.no_report
    )


if __name__ == "__main__":
    main()
