#!/usr/bin/env python3
"""
Multi-experiment runner for hyperparameter sweeps.

Usage:
    python run_multi_experiments.py --sweep_config config/multi_experiments.yaml
"""

import os
import yaml
import json
import subprocess
import argparse
import itertools
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Tuple
from tqdm import tqdm # <--- Added for progress bar

def set_nested_value(d: Dict, key_path: str, value: Any):
    """Set a value in a nested dictionary using dot notation."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_nested_value(d: Dict, key_path: str, default=None):
    """Get a value from a nested dictionary using dot notation."""
    keys = key_path.split('.')
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def check_constraint(config: Dict, constraint: Dict) -> bool:
    """Check if a constraint condition is satisfied."""
    condition = constraint.get('condition', '')
    
    # Parse simple conditions like "recursion.recursive_mode == false"
    if '==' in condition:
        key_path, value_str = condition.split('==')
        key_path = key_path.strip()
        value_str = value_str.strip().lower()
        
        actual_value = get_nested_value(config, key_path)
        
        # Convert string representation to actual value
        if value_str == 'true':
            expected_value = True
        elif value_str == 'false':
            expected_value = False
        else:
            try:
                expected_value = int(value_str)
            except ValueError:
                try:
                    expected_value = float(value_str)
                except ValueError:
                    expected_value = value_str.strip('"\'')
        
        return actual_value == expected_value
    
    return False

def apply_constraint(config: Dict, constraint: Dict):
    """Apply constraint settings to config."""
    if 'set' in constraint:
        for key_path, value in constraint['set'].items():
            set_nested_value(config, key_path, value)

def generate_experiment_configs(sweep_config: Dict) -> List[Tuple[Dict, str]]:
    """Generate all experiment configurations from sweep parameters."""
    base_config_path = sweep_config.get('base_config', 'config/config.yaml')
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Get sweep parameters
    sweep_params = sweep_config.get('sweep', {})
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    configs = []
    for idx, values in enumerate(itertools.product(*param_values)):
        # Create a copy of base config
        # Use deepcopy pattern for safety, though yaml.safe_load(yaml.dump(base_config)) also works
        config = yaml.full_load(yaml.dump(base_config)) 
        
        # Apply sweep parameters
        param_dict = {}
        for param_name, value in zip(param_names, values):
            set_nested_value(config, param_name, value)
            param_dict[param_name] = value
        
        # Apply constraints
        constraints = sweep_config.get('constraints', [])
        for constraint in constraints:
            if check_constraint(config, constraint):
                apply_constraint(config, constraint)
                # Update param_dict with constraint values
                if 'set' in constraint:
                    for key_path, value in constraint['set'].items():
                        param_dict[key_path] = value
        
        # Generate experiment name
        exp_name_parts = []
        for param_name, value in param_dict.items():
            short_name = param_name.split('.')[-1]
            # Convert boolean to string for name
            if isinstance(value, bool):
                 value_str = str(value).lower()
            else:
                 value_str = str(value)
            exp_name_parts.append(f"{short_name}={value_str}")
        exp_name = "_".join(exp_name_parts)
        exp_name = f"exp_{idx:03d}_{exp_name}"
        
        configs.append((config, exp_name))
    
    return configs

def save_experiment_config(config: Dict, output_dir: str, exp_name: str) -> str:
    """Save experiment config to a temporary file."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{exp_name}_config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def run_experiment(config_path: str, cuda_device: str, nproc: int = 1) -> Tuple[bool, str]:
    """
    Run a single experiment using torchrun, suppressing stdout/stderr
    to prevent logs from cluttering the sweep output.
    """
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "src/train.py",
        "--config", config_path
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    # print(f"Running command: {' '.join(cmd)}") # Suppressing verbose output
    # print(f"CUDA_VISIBLE_DEVICES={cuda_device}") # Suppressing verbose output
    
    try:
        # Redirect stdout and stderr to DEVNULL to suppress logs from src/train.py
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        # Note: result.stdout and result.stderr are now empty/logged in a file 
        # inside the experiment directory by the inner script if it logs correctly.
        return True, "" # Return empty string for output as it's suppressed
    except subprocess.CalledProcessError as e:
        # If an error occurs, print the error output for debugging
        print(f"Error running experiment:")
        print(f"Stdout:\n{e.stdout}") 
        print(f"Stderr:\n{e.stderr}")
        return False, e.stderr

def collect_results(experiments_dir: str) -> pd.DataFrame:
    """Collect results from all experiments."""
    # ... (function body remains the same as it handles post-run analysis)
    results = []
    
    for exp_dir in sorted(Path(experiments_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Load test results
        test_results_path = exp_dir / "test_results.json"
        if not test_results_path.exists():
            # Only print warning if it looks like an experiment directory (starts with exp_)
            if exp_dir.name.startswith("exp_"):
                 print(f"Warning: No test_results.json found in {exp_dir.name}")
            continue
        
        with open(test_results_path, 'r') as f:
            try:
                test_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode test_results.json in {exp_dir.name}")
                continue
        
        # Load config
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.full_load(f)
            
            result = {
                'experiment_name': exp_dir.name,
                'dataset': get_nested_value(config, 'data.dataset_name', 'N/A'),
                'recursive_mode': get_nested_value(config, 'recursion.recursive_mode', 'N/A'),
                'N_supervision_steps': get_nested_value(config, 'recursion.N_supervision_steps', 'N/A'),
                'N_latent_steps': get_nested_value(config, 'recursion.N_latent_steps', 'N/A'),
                'N_deep_steps': get_nested_value(config, 'recursion.N_deep_steps', 'N/A'),
                'test_N_supervision_steps': get_nested_value(config, 'testing.N_supervision_steps', 'N/A'),
                'test_accuracy': test_results.get('test_accuracy'),
                'test_loss': test_results.get('test_loss'),
            }
            results.append(result)
    
    return pd.DataFrame(results)

def create_visualizations(df: pd.DataFrame, output_dir: str):
    # ... (function body remains the same, no changes needed here)
    """Create visualization plots for the multi-experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Accuracy comparison across datasets
    if len(df['dataset'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_grouped = df.groupby('dataset')['test_accuracy'].agg(['mean', 'std', 'max'])
        df_grouped = df_grouped.sort_values('mean', ascending=False)
        
        x = range(len(df_grouped))
        ax.bar(x, df_grouped['mean'], yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Test Accuracy by Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_grouped.index, rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_dataset.png'), dpi=300)
        plt.close()
    
    # 2. Recursive vs Non-Recursive comparison
    if df['recursive_mode'].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Ensure recursive_mode is a boolean or convertible
        df_plot = df.copy()
        df_plot['recursive_mode'] = df_plot['recursive_mode'].astype(bool)
        
        for dataset in df_plot['dataset'].unique():
            df_dataset = df_plot[df_plot['dataset'] == dataset]
            
            recursive_acc = df_dataset[df_dataset['recursive_mode'] == True]['test_accuracy'].values
            non_recursive_acc = df_dataset[df_dataset['recursive_mode'] == False]['test_accuracy'].values
            
            if len(recursive_acc) > 0 and len(non_recursive_acc) > 0:
                axes[0].scatter([dataset]*len(recursive_acc), recursive_acc, 
                                 label=f'{dataset} (Recursive)', alpha=0.6, s=100)
                axes[0].scatter([dataset]*len(non_recursive_acc), non_recursive_acc, 
                                 label=f'{dataset} (Non-Recursive)', alpha=0.6, s=100, marker='x')
        
        axes[0].set_xlabel('Dataset', fontsize=12)
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_title('Recursive vs Non-Recursive Mode', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot comparison
        df_melted = df_plot[['recursive_mode', 'test_accuracy']].copy()
        df_melted['recursive_mode'] = df_melted['recursive_mode'].map({True: 'Recursive', False: 'Non-Recursive'})
        sns.boxplot(data=df_melted, x='recursive_mode', y='test_accuracy', ax=axes[1])
        axes[1].set_xlabel('Mode', fontsize=12)
        axes[1].set_ylabel('Test Accuracy', fontsize=12)
        axes[1].set_title('Accuracy Distribution by Mode', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recursive_comparison.png'), dpi=300)
        plt.close()
    
    # 3. Hyperparameter effects (for recursive mode only)
    df_recursive = df[df['recursive_mode'] == True]
    
    if len(df_recursive) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # N_supervision_steps effect
        if df_recursive['N_supervision_steps'].nunique() > 1:
            for dataset in df_recursive['dataset'].unique():
                df_dataset = df_recursive[df_recursive['dataset'] == dataset]
                grouped = df_dataset.groupby('N_supervision_steps')['test_accuracy'].mean()
                axes[0, 0].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
            
            axes[0, 0].set_xlabel('N_supervision_steps', fontsize=12)
            axes[0, 0].set_ylabel('Test Accuracy', fontsize=12)
            axes[0, 0].set_title('Effect of N_supervision_steps', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # N_latent_steps effect
        if df_recursive['N_latent_steps'].nunique() > 1:
            for dataset in df_recursive['dataset'].unique():
                df_dataset = df_recursive[df_recursive['dataset'] == dataset]
                grouped = df_dataset.groupby('N_latent_steps')['test_accuracy'].mean()
                axes[0, 1].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
            
            axes[0, 1].set_xlabel('N_latent_steps', fontsize=12)
            axes[0, 1].set_ylabel('Test Accuracy', fontsize=12)
            axes[0, 1].set_title('Effect of N_latent_steps', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # N_deep_steps effect
        if df_recursive['N_deep_steps'].nunique() > 1:
            for dataset in df_recursive['dataset'].unique():
                df_dataset = df_recursive[df_recursive['dataset'] == dataset]
                grouped = df_dataset.groupby('N_deep_steps')['test_accuracy'].mean()
                axes[1, 0].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
            
            axes[1, 0].set_xlabel('N_deep_steps', fontsize=12)
            axes[1, 0].set_ylabel('Test Accuracy', fontsize=12)
            axes[1, 0].set_title('Effect of N_deep_steps', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        # Test N_supervision_steps effect
        if df_recursive['test_N_supervision_steps'].nunique() > 1:
            for dataset in df_recursive['dataset'].unique():
                df_dataset = df_recursive[df_recursive['dataset'] == dataset]
                grouped = df_dataset.groupby('test_N_supervision_steps')['test_accuracy'].mean()
                axes[1, 1].plot(grouped.index, grouped.values, marker='o', label=dataset, linewidth=2)
            
            axes[1, 1].set_xlabel('Test N_supervision_steps', fontsize=12)
            axes[1, 1].set_ylabel('Test Accuracy', fontsize=12)
            axes[1, 1].set_title('Effect of Test N_supervision_steps', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hyperparameter_effects.png'), dpi=300)
        plt.close()
    
    # 4. Heatmap of accuracy (if applicable)
    if len(df_recursive) > 1 and df_recursive['N_supervision_steps'].nunique() > 1 and df_recursive['N_latent_steps'].nunique() > 1:
        for dataset in df_recursive['dataset'].unique():
            df_dataset = df_recursive[df_recursive['dataset'] == dataset]
            
            pivot_table = df_dataset.pivot_table(
                values='test_accuracy',
                index='N_supervision_steps',
                columns='N_latent_steps',
                aggfunc='mean'
            )
            
            if not pivot_table.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, 
                            cbar_kws={'label': 'Test Accuracy'})
                ax.set_title(f'Accuracy Heatmap: {dataset}\n(N_supervision_steps vs N_latent_steps)', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('N_latent_steps', fontsize=12)
                ax.set_ylabel('N_supervision_steps', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'heatmap_{dataset}.png'), dpi=300)
                plt.close()
    
    # 5. Top performers table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    top_10 = df.nlargest(10, 'test_accuracy')[['experiment_name', 'dataset', 'recursive_mode', 
                                                   'N_supervision_steps', 'N_latent_steps', 
                                                   'N_deep_steps', 'test_accuracy']]
    top_10['test_accuracy'] = top_10['test_accuracy'].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=top_10.values, colLabels=top_10.columns,
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(top_10.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Top 10 Experiments by Test Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'top_performers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with different hyperparameters")
    parser.add_argument(
        "--sweep_config",
        type=str,
        default="config/multi_experiments.yaml",
        help="Path to the multi-experiment sweep configuration file"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes per node for torchrun"
    )
    args = parser.parse_args()
    
    # Load sweep configuration
    if not os.path.exists(args.sweep_config):
        print(f"Error: Sweep config file not found at {args.sweep_config}")
        exit(1)
    
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.full_load(f)
    
    print(f"Loaded sweep configuration from {args.sweep_config}")
    print(f"Description: {sweep_config.get('description', 'N/A')}")
    
    # Generate experiment configurations
    experiment_configs = generate_experiment_configs(sweep_config)
    num_experiments = len(experiment_configs)
    print(f"\nGenerated {num_experiments} experiment configurations")
    
    # Create multi-experiments directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    multi_exp_dir = os.path.join("experiments", "multi_experiments", f"sweep_{timestamp}")
    os.makedirs(multi_exp_dir, exist_ok=True)
    
    # Save sweep configuration
    with open(os.path.join(multi_exp_dir, "sweep_config.yaml"), 'w') as f:
        yaml.dump(sweep_config, f)
    
    # Get experiment settings
    exp_settings = sweep_config.get('experiment_settings', {})
    cuda_devices = exp_settings.get('cuda_devices', ['0'])
    continue_on_error = exp_settings.get('continue_on_error', True)
    
    # Run experiments
    successful_experiments = []
    failed_experiments = []
    
    # Initialize progress bar for the main process (rank 0)
    # The progress bar is only needed in the main script running the sweep.
    # We use dynamic description to show the current experiment being run.
    with tqdm(total=num_experiments, desc="Total Sweep Progress", unit="exp") as pbar:
        for idx, (config, exp_name) in enumerate(experiment_configs):
            
            # Update progress bar description to show current experiment
            pbar.set_description(f"Sweep Progress (Running: {exp_name})")

            # Save experiment config
            config_path = save_experiment_config(config, multi_exp_dir, exp_name)
            
            # Select CUDA device (round-robin)
            cuda_device = cuda_devices[idx % len(cuda_devices)]
            
            # Run experiment
            # The run_experiment function now suppresses logs
            success, output = run_experiment(config_path, cuda_device, args.nproc)
            
            if success:
                successful_experiments.append(exp_name)
            else:
                failed_experiments.append(exp_name)
                # Print error details (already done in run_experiment, but good to mark the failure here)
                print(f"✗ Experiment {exp_name} failed. Check error output above.")
                if not continue_on_error:
                    print("Stopping due to experiment failure (continue_on_error=False)")
                    break
            
            # Update progress bar
            pbar.update(1)

        # Reset progress bar description after loop finishes
        pbar.set_description("Sweep Progress (Completed)")
    
    # Collect and analyze results
    print(f"\n{'='*80}")
    print("Multi-experiment sweep completed!")
    print(f"{'='*80}")
    print(f"Successful experiments: {len(successful_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp_name in failed_experiments:
            print(f"  - {exp_name}")
    
    # Collect results from experiments directory
    experiments_base_dir = sweep_config.get('base_config', 'config/config.yaml')
    try:
        with open(experiments_base_dir, 'r') as f:
            base_config = yaml.safe_load(f)
        experiments_dir = base_config['saving']['base_output_dir']
    except Exception as e:
        print(f"Could not load base config to find results directory: {e}")
        return

    
    print(f"\nCollecting results from {experiments_dir}...")
    df = collect_results(experiments_dir)
    
    if not df.empty:
        # Save results summary
        results_csv_path = os.path.join(multi_exp_dir, "results_summary.csv")
        df.to_csv(results_csv_path, index=False)
        print(f"Results summary saved to {results_csv_path}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(df, multi_exp_dir)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\nBest accuracy: {df['test_accuracy'].max():.4f}")
        best_exp = df.loc[df['test_accuracy'].idxmax()]
        print(f"Best experiment: {best_exp['experiment_name']}")
        print(f"  Dataset: {best_exp['dataset']}")
        print(f"  Recursive mode: {best_exp['recursive_mode']}")
        print(f"  N_supervision_steps: {best_exp['N_supervision_steps']}")
        print(f"  N_latent_steps: {best_exp['N_latent_steps']}")
        print(f"  N_deep_steps: {best_exp['N_deep_steps']}")
        
        print(f"\nAverage accuracy: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_experiments': num_experiments,
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(failed_experiments),
            'best_accuracy': float(df['test_accuracy'].max()),
            'best_experiment': best_exp.to_dict(),
            'average_accuracy': float(df['test_accuracy'].mean()),
            'std_accuracy': float(df['test_accuracy'].std())
        }
        
        with open(os.path.join(multi_exp_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("No results collected. Check if experiments completed successfully and saved 'test_results.json'.")
    
    print(f"\nAll results saved to: {multi_exp_dir}")

if __name__ == "__main__":
    main()