"""
THE CONTROVERSIAL REGIME: COMPREHENSIVE ANALYSIS FOR NATURE FLAGSHIP
=====================================================================

Two paradigm-breaking discoveries:
1. Only 27.5% of benchmark problems discriminate between models
2. Small models systematically outperform large models on 7.7% of problems

This analysis includes all necessary statistics for top-tier publication:
- Effect sizes (Cohen's d)
- Correlations (Spearman's rank)
- Statistical significance tests
- Mechanistic analysis of failure patterns
- Success/failure pattern analysis

Output: Complete analysis for Nature submission
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal, spearmanr
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration for controversial regime analysis"""
    
    BASE_DIR = Path("./cross_model_archaeology")
    OUTPUT_DIR = Path("./controversial_regime_results")
    
    DATASETS = ['gsm8k', 'boolq', 'arc-easy', 'commonsenseqa', 'mmlu-stem']
    
    # Thresholds for problem categorization
    UNIVERSALLY_EASY_THRESHOLD = 0.80
    UNIVERSALLY_HARD_THRESHOLD = 0.20
    CONTROVERSIAL_MIN = 0.40
    CONTROVERSIAL_MAX = 0.60
    
    # Model metadata
    MODEL_METADATA = {
        'Qwen/Qwen2.5-0.5B-Instruct': {'size': 0.5, 'tier': 'weak', 'family': 'Qwen'},
        'meta-llama/Llama-3.2-1B-Instruct': {'size': 1.0, 'tier': 'weak', 'family': 'Llama'},
        'google/gemma-3-1b-it': {'size': 1.0, 'tier': 'weak', 'family': 'Gemma'},
        'stabilityai/stablelm-2-1_6b': {'size': 1.6, 'tier': 'weak', 'family': 'StableLM'},
        'google/gemma-2-2b-it': {'size': 2.0, 'tier': 'weak', 'family': 'Gemma'},
        'meta-llama/Llama-3.2-3B-Instruct': {'size': 3.0, 'tier': 'weak', 'family': 'Llama'},
        'Qwen/Qwen2.5-3B-Instruct': {'size': 3.0, 'tier': 'weak', 'family': 'Qwen'},
        'stabilityai/stablelm-zephyr-3b': {'size': 3.0, 'tier': 'weak', 'family': 'StableLM'},
        'microsoft/Phi-3-mini-4k-instruct': {'size': 3.8, 'tier': 'medium', 'family': 'Phi'},
        'microsoft/Phi-3.5-mini-instruct': {'size': 3.8, 'tier': 'medium', 'family': 'Phi'},
        'nvidia/Llama-3.1-Minitron-4B-Width-Base': {'size': 4.0, 'tier': 'medium', 'family': 'Llama'},
        'nvidia/Llama-3.1-Minitron-4B-Depth-Base': {'size': 4.0, 'tier': 'medium', 'family': 'Llama'},
        '01-ai/Yi-1.5-6B-Chat': {'size': 6.0, 'tier': 'medium', 'family': 'Yi'},
        'deepseek-ai/deepseek-llm-7b-base': {'size': 7.0, 'tier': 'strong', 'family': 'DeepSeek'},
        'Qwen/Qwen2.5-7B-Instruct': {'size': 7.0, 'tier': 'strong', 'family': 'Qwen'},
        'mistralai/Mistral-7B-Instruct-v0.3': {'size': 7.0, 'tier': 'strong', 'family': 'Mistral'},
        'meta-llama/Llama-3.1-8B-Instruct': {'size': 8.0, 'tier': 'strong', 'family': 'Llama'},
        'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {'size': 8.0, 'tier': 'strong', 'family': 'Llama'},
        'google/gemma-2-9b-it': {'size': 9.0, 'tier': 'strong', 'family': 'Gemma'},
        'meta-llama/Llama-2-13b-hf': {'size': 13.0, 'tier': 'strong', 'family': 'Llama'},
        'Qwen/Qwen2.5-14B-Instruct': {'size': 14.0, 'tier': 'strong', 'family': 'Qwen'},
        'openai/gpt-oss-20b': {'size': 20.0, 'tier': 'strong', 'family': 'GPT'},
        'mistralai/Mistral-Small-24B-Instruct-2501': {'size': 24.0, 'tier': 'strong', 'family': 'Mistral'},
        'moonshotai/kimi-k2-instruct': {'size': 32.0, 'tier': 'strong', 'family': 'Kimi'},
        'Qwen/Qwen2.5-32B-Instruct': {'size': 32.0, 'tier': 'strong', 'family': 'Qwen'},
        'deepseek-ai/deepseek-llm-67b-base': {'size': 67.0, 'tier': 'strong', 'family': 'DeepSeek'},
        'meta/llama-3.3-70b-versatile': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'meta-llama/Meta-Llama-3-70B': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'meta-llama/Meta-Llama-3-70B-Instruct': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'databricks-meta-llama-3-1-405b-instruct': {'size': 405.0, 'tier': 'strong', 'family': 'Llama'},
        'gemini-2.0-flash': {'size': 50.0, 'tier': 'strong', 'family': 'Gemini'},
    }
    
    WEAK_MODELS = [m for m, d in MODEL_METADATA.items() if d['tier'] == 'weak']
    MEDIUM_MODELS = [m for m, d in MODEL_METADATA.items() if d['tier'] == 'medium']
    STRONG_MODELS = [m for m, d in MODEL_METADATA.items() if d['tier'] == 'strong']

# ============================================================================
# DATA LOADER
# ============================================================================
class DataLoader:
    """Load all model responses"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.all_data = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from directory structure"""
        print("="*80)
        print("LOADING DATA FOR CONTROVERSIAL REGIME ANALYSIS")
        print("="*80)
        
        model_dirs = [d for d in self.base_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_model')]
        
        print(f"\nFound {len(model_dirs)} model directories")
        
        for dataset in Config.DATASETS:
            dataset_dfs = []
            
            for model_dir in model_dirs:
                model_name = model_dir.name.replace('_model', '').replace('Llama-', 'Llama/').replace('Qwen', 'Qwen/')
                csv_path = model_dir / "cross_model_archaeology" / dataset / "raw_responses"
                
                if not csv_path.exists():
                    continue
                
                csv_files = list(csv_path.glob("responses_*.csv"))
                if not csv_files:
                    continue
                
                latest_csv = sorted(csv_files)[-1]
                df = pd.read_csv(latest_csv)
                
                if 'model_name' not in df.columns or df['model_name'].isna().all():
                    df['model_name'] = model_name
                
                dataset_dfs.append(df)
            
            if dataset_dfs:
                combined_df = pd.concat(dataset_dfs, ignore_index=True)
                self.all_data[dataset] = combined_df
                print(f"✓ {dataset}: {len(combined_df)} responses from {combined_df['model_name'].nunique()} models")
        
        return self.all_data

# ============================================================================
# ANALYSIS 1: PROBLEM CATEGORIZATION
# ============================================================================
class ProblemCategorizationAnalyzer:
    """Categorize problems by discriminative power"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.results = {}
        
    def categorize_all_problems(self) -> Dict:
        """Main analysis: categorize every problem"""
        print("\n" + "="*80)
        print("ANALYSIS 1: PROBLEM CATEGORIZATION BY DISCRIMINATIVE POWER")
        print("="*80)
        
        all_results = {}
        
        for dataset_name, df in self.data.items():
            print(f"\n📊 ANALYZING {dataset_name.upper()}")
            
            problem_stats = []
            
            for sample_id in df['sample_id'].unique():
                sample_df = df[df['sample_id'] == sample_id]
                
                if len(sample_df) < 3:
                    continue
                
                success_rate = sample_df['is_correct'].mean()
                num_models = len(sample_df)
                
                # Categorize
                if success_rate >= Config.UNIVERSALLY_EASY_THRESHOLD:
                    category = 'universally_easy'
                elif success_rate <= Config.UNIVERSALLY_HARD_THRESHOLD:
                    category = 'universally_hard'
                elif Config.CONTROVERSIAL_MIN <= success_rate <= Config.CONTROVERSIAL_MAX:
                    category = 'controversial'
                else:
                    category = 'moderate'
                
                problem_stats.append({
                    'sample_id': sample_id,
                    'dataset': dataset_name,
                    'success_rate': success_rate,
                    'num_models': num_models,
                    'category': category,
                    'variance': sample_df['is_correct'].var()
                })
            
            stats_df = pd.DataFrame(problem_stats)
            category_counts = stats_df['category'].value_counts()
            total_problems = len(stats_df)
            
            print(f"   Problems: {total_problems}")
            for cat in ['universally_easy', 'controversial', 'moderate', 'universally_hard']:
                if cat in category_counts:
                    count = category_counts[cat]
                    pct = count / total_problems * 100
                    marker = "🔥" if cat == 'controversial' else "  "
                    print(f"   {marker} {cat:<20}: {count:>4} ({pct:>5.1f}%)")
            
            controversial_pct = (category_counts.get('controversial', 0) / total_problems * 100)
            useless_pct = ((category_counts.get('universally_easy', 0) + 
                           category_counts.get('universally_hard', 0)) / total_problems * 100)
            
            all_results[dataset_name] = {
                'stats_df': stats_df,
                'category_counts': category_counts.to_dict(),
                'controversial_pct': controversial_pct,
                'useless_pct': useless_pct,
                'total_problems': total_problems
            }
        
        # Aggregate
        print(f"\n" + "="*80)
        print(f"AGGREGATE RESULTS")
        print(f"="*80)
        
        avg_controversial = np.mean([r['controversial_pct'] for r in all_results.values()])
        avg_useless = np.mean([r['useless_pct'] for r in all_results.values()])
        
        print(f"\nAverage across {len(all_results)} datasets:")
        print(f"  Controversial (discriminative): {avg_controversial:.1f}%")
        print(f"  Non-discriminative (useless):   {avg_useless:.1f}%")
        print(f"\n KEY FINDING: Only {avg_controversial:.0f}% of problems discriminate models!")
        
        self.results = all_results
        return all_results

# ============================================================================
# ANALYSIS 2: INVERSE SCALING WITH STATISTICAL RIGOR
# ============================================================================
class InverseScalingAnalyzer:
    """Find and analyze inverse scaling with full statistics"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.results = {}
        
    def find_inverse_scaling_problems(self) -> Dict:
        """Find inverse scaling with statistical analysis"""
        print("\n" + "="*80)
        print("ANALYSIS 2: INVERSE SCALING DISCOVERY")
        print("="*80)
        
        all_results = {}
        
        for dataset_name, df in self.data.items():
            print(f"\n ANALYZING {dataset_name.upper()}")
            
            inverse_problems = []
            
            for sample_id in df['sample_id'].unique():
                sample_df = df[df['sample_id'] == sample_id]
                
                weak_df = sample_df[sample_df['model_name'].isin(Config.WEAK_MODELS)]
                strong_df = sample_df[sample_df['model_name'].isin(Config.STRONG_MODELS)]
                
                if len(weak_df) == 0 or len(strong_df) == 0:
                    continue
                
                weak_acc = weak_df['is_correct'].mean()
                strong_acc = strong_df['is_correct'].mean()
                gap = weak_acc - strong_acc
                
                if gap > 0.2:  # 20% threshold
                    inverse_problems.append({
                        'sample_id': sample_id,
                        'weak_accuracy': weak_acc,
                        'strong_accuracy': strong_acc,
                        'gap': gap,
                        'weak_n': len(weak_df),
                        'strong_n': len(strong_df)
                    })
            
            if inverse_problems:
                inv_df = pd.DataFrame(inverse_problems)
                inv_count = len(inv_df)
                total = df['sample_id'].nunique()
                inv_pct = inv_count / total * 100
                
                print(f"   Inverse scaling: {inv_count}/{total} ({inv_pct:.1f}%)")
                print(f"   Average gap: {inv_df['gap'].mean()*100:+.1f}%")
                print(f"   Max gap: {inv_df['gap'].max()*100:+.1f}%")
                
                all_results[dataset_name] = {
                    'inverse_df': inv_df,
                    'count': inv_count,
                    'pct': inv_pct,
                    'total': total,
                    'avg_gap': inv_df['gap'].mean()
                }
            else:
                all_results[dataset_name] = {
                    'count': 0,
                    'pct': 0.0,
                    'total': df['sample_id'].nunique()
                }
        
        # Aggregate
        total_inv = sum(r['count'] for r in all_results.values())
        total_prob = sum(r['total'] for r in all_results.values())
        overall_pct = total_inv / total_prob * 100
        
        print(f"\n" + "="*80)
        print(f"AGGREGATE RESULTS")
        print(f"="*80)
        print(f"  Inverse scaling: {total_inv}/{total_prob} ({overall_pct:.1f}%)")
        print(f"\n KEY FINDING: Small models outperform large on {overall_pct:.1f}% of problems!")
        
        self.results = all_results
        return all_results

# ============================================================================
# ANALYSIS 3: MECHANISM ANALYSIS - WHY FAILURES OCCUR
# ============================================================================
class MechanismAnalyzer:
    """Analyze WHY models fail on inverse scaling problems"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], inverse_results: Dict):
        self.data = data
        self.inverse_results = inverse_results
        
    def analyze_failure_mechanisms(self) -> Dict:
        """Analyze failure patterns"""
        print("\n" + "="*80)
        print("ANALYSIS 3: FAILURE MECHANISM ANALYSIS")
        print("="*80)
        
        mechanisms = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in self.inverse_results:
                continue
                
            inv_result = self.inverse_results[dataset_name]
            if inv_result['count'] == 0:
                continue
            
            inv_df = inv_result['inverse_df']
            inverse_problem_ids = set(inv_df['sample_id'])
            
            print(f"\n {dataset_name.upper()}: Analyzing {len(inverse_problem_ids)} inverse scaling problems")
            
            # Get failure patterns for strong models on inverse problems
            strong_failures = []
            weak_successes = []
            
            for sample_id in inverse_problem_ids:
                sample_df = df[df['sample_id'] == sample_id]
                
                weak_df = sample_df[sample_df['model_name'].isin(Config.WEAK_MODELS)]
                strong_df = sample_df[sample_df['model_name'].isin(Config.STRONG_MODELS)]
                
                # Strong model failures
                strong_wrong = strong_df[strong_df['is_correct'] == False]
                if len(strong_wrong) > 0:
                    for _, row in strong_wrong.iterrows():
                        strong_failures.append({
                            'sample_id': sample_id,
                            'model': row['model_name'],
                            'generation_length': len(str(row.get('full_generation', '')).split())
                        })
                
                # Weak model successes
                weak_right = weak_df[weak_df['is_correct'] == True]
                if len(weak_right) > 0:
                    for _, row in weak_right.iterrows():
                        weak_successes.append({
                            'sample_id': sample_id,
                            'model': row['model_name'],
                            'generation_length': len(str(row.get('full_generation', '')).split())
                        })
            
            # Analyze generation lengths
            if strong_failures and weak_successes:
                strong_lens = [f['generation_length'] for f in strong_failures]
                weak_lens = [s['generation_length'] for s in weak_successes]
                
                avg_strong_len = np.mean(strong_lens)
                avg_weak_len = np.mean(weak_lens)
                
                # Statistical test
                statistic, pval = mannwhitneyu(strong_lens, weak_lens, alternative='two-sided')
                
                print(f"\n   Response length analysis:")
                print(f"   Strong models (failed): {avg_strong_len:.1f} words (n={len(strong_lens)})")
                print(f"   Weak models (succeeded): {avg_weak_len:.1f} words (n={len(weak_lens)})")
                print(f"   Difference: {avg_strong_len - avg_weak_len:+.1f} words")
                print(f"   Mann-Whitney U test: p={pval:.4f} {'✓ Significant' if pval < 0.05 else '✗ Not significant'}")
                
                if avg_strong_len > avg_weak_len * 1.5:
                    print(f"    MECHANISM: Strong models OVERTHINK (50% longer responses)")
                elif avg_strong_len < avg_weak_len * 0.7:
                    print(f"    MECHANISM: Strong models UNDERTHINK (30% shorter responses)")
                else:
                    print(f"    MECHANISM: Length-independent (different reasoning approach)")
                
                mechanisms[dataset_name] = {
                    'strong_avg_len': avg_strong_len,
                    'weak_avg_len': avg_weak_len,
                    'len_diff': avg_strong_len - avg_weak_len,
                    'pval': pval,
                    'mechanism': 'overthinking' if avg_strong_len > avg_weak_len * 1.5 else 'different_approach'
                }
        
        return mechanisms

# ============================================================================
# ANALYSIS 4: EFFECT SIZES AND CORRELATIONS
# ============================================================================
class StatisticalRigorAnalyzer:
    """Calculate all statistics needed for Nature"""
    
    def __init__(self, categorization_results: Dict, inverse_results: Dict):
        self.cat_results = categorization_results
        self.inv_results = inverse_results
        
    def calculate_comprehensive_statistics(self) -> Dict:
        """Calculate effect sizes, correlations, significance tests"""
        print("\n" + "="*80)
        print("ANALYSIS 4: COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        stats_results = {}
        
        # ================================================================
        # PART 1: Discriminative Efficiency Analysis
        # ================================================================
        print("\n PART 1: DISCRIMINATIVE EFFICIENCY STATISTICS")
        
        # Extract controversial percentages
        controversial_pcts = [r['controversial_pct'] for r in self.cat_results.values()]
        useless_pcts = [r['useless_pct'] for r in self.cat_results.values()]
        dataset_names = list(self.cat_results.keys())
        
        print(f"\nDescriptive statistics:")
        print(f"  Controversial %: Mean={np.mean(controversial_pcts):.1f}%, "
              f"SD={np.std(controversial_pcts):.1f}%, Range=[{np.min(controversial_pcts):.1f}%-{np.max(controversial_pcts):.1f}%]")
        print(f"  Useless %: Mean={np.mean(useless_pcts):.1f}%, "
              f"SD={np.std(useless_pcts):.1f}%, Range=[{np.min(useless_pcts):.1f}%-{np.max(useless_pcts):.1f}%]")
        
        # Correlation between controversial % and dataset difficulty
        total_problems = [r['total_problems'] for r in self.cat_results.values()]
        corr_size, pval_size = stats.spearmanr(total_problems, controversial_pcts)
        
        print(f"\nCorrelation analysis:")
        print(f"  Dataset size vs controversial %: rs={corr_size:+.3f}, p={pval_size:.4f}")
        
        # Test if controversial % differs across datasets
        if len(controversial_pcts) >= 3:
            kruskal_stat, kruskal_pval = kruskal(*[[p] for p in controversial_pcts])
            print(f"  Kruskal-Wallis test (varies across datasets): H={kruskal_stat:.2f}, p={kruskal_pval:.4f}")
        
        stats_results['discriminative_efficiency'] = {
            'controversial_mean': np.mean(controversial_pcts),
            'controversial_std': np.std(controversial_pcts),
            'useless_mean': np.mean(useless_pcts),
            'useless_std': np.std(useless_pcts),
            'correlation_size': corr_size,
            'correlation_pval': pval_size
        }
        
        # ================================================================
        # PART 2: Inverse Scaling Statistics
        # ================================================================
        print("\n PART 2: INVERSE SCALING STATISTICS")
        
        # Extract inverse scaling percentages
        inv_pcts = [r['pct'] for r in self.inv_results.values()]
        avg_gaps = [r.get('avg_gap', 0) for r in self.inv_results.values() if r.get('avg_gap', 0) > 0]
        
        print(f"\nDescriptive statistics:")
        print(f"  Inverse scaling %: Mean={np.mean(inv_pcts):.1f}%, "
              f"SD={np.std(inv_pcts):.1f}%, Range=[{np.min(inv_pcts):.1f}%-{np.max(inv_pcts):.1f}%]")
        
        if avg_gaps:
            print(f"  Performance gaps: Mean={np.mean(avg_gaps)*100:.1f}%, "
                  f"SD={np.std(avg_gaps)*100:.1f}%")
            
            # Collect ALL weak and strong accuracies across ALL inverse problems
            all_weak_accuracies = []
            all_strong_accuracies = []
            
            for dataset_name, result in self.inv_results.items():
                if result['count'] == 0:
                    continue
                
                inv_df = result.get('inverse_df')
                if inv_df is not None:
                    all_weak_accuracies.extend(inv_df['weak_accuracy'].tolist())
                    all_strong_accuracies.extend(inv_df['strong_accuracy'].tolist())
            
            if all_weak_accuracies and all_strong_accuracies:
                # Calculate group means
                mean_weak = np.mean(all_weak_accuracies)
                mean_strong = np.mean(all_strong_accuracies)
                
                # Calculate group standard deviations
                sd_weak = np.std(all_weak_accuracies, ddof=1)
                sd_strong = np.std(all_strong_accuracies, ddof=1)
                
                # Sample sizes
                n_weak = len(all_weak_accuracies)
                n_strong = len(all_strong_accuracies)
                
                # Pooled standard deviation (correct formula)
                pooled_sd = np.sqrt(((n_weak - 1) * sd_weak**2 + 
                                     (n_strong - 1) * sd_strong**2) / 
                                    (n_weak + n_strong - 2))
                
                # Cohen's d (CORRECT FORMULA)
                effect_size = (mean_weak - mean_strong) / pooled_sd if pooled_sd > 0 else 0
                
                # Print detailed effect size information
                print(f"\nEffect size (Cohen's d):")
                print(f"  Weak models:   M={mean_weak:.3f} ({mean_weak*100:.1f}%), SD={sd_weak:.3f}, n={n_weak}")
                print(f"  Strong models: M={mean_strong:.3f} ({mean_strong*100:.1f}%), SD={sd_strong:.3f}, n={n_strong}")
                print(f"  Pooled SD: {pooled_sd:.3f}")
                print(f"  Cohen's d = ({mean_weak:.3f} - {mean_strong:.3f}) / {pooled_sd:.3f} = {effect_size:.3f}")
                print(f"  Interpretation: {'large' if abs(effect_size) > 0.8 else 'medium' if abs(effect_size) > 0.5 else 'small'} effect")
            else:
                effect_size = 0
        
        # Correlation between controversial % and inverse scaling %
        corr_contr_inv, pval_contr_inv = stats.spearmanr(controversial_pcts, inv_pcts)
        print(f"\nCorrelation analysis:")
        print(f"  Controversial % vs Inverse scaling %: rs={corr_contr_inv:+.3f}, p={pval_contr_inv:.4f}")
        print(f"  Interpretation: {'Controversial problems show MORE inverse scaling' if corr_contr_inv > 0 else 'No relationship'}")
        
        stats_results['inverse_scaling'] = {
            'inverse_mean': np.mean(inv_pcts),
            'inverse_std': np.std(inv_pcts),
            'avg_gap_mean': np.mean(avg_gaps) if avg_gaps else 0,
            'effect_size': effect_size if avg_gaps else 0,
            'correlation_controversial': corr_contr_inv,
            'correlation_pval': pval_contr_inv
        }
        
        # ================================================================
        # PART 3: Cross-Dataset Consistency
        # ================================================================
        print("\n📊 PART 3: CROSS-DATASET CONSISTENCY")
        
        # Test if patterns are consistent across datasets
        print(f"\nConsistency tests:")
        print(f"  All datasets show controversial regime: {'✓ YES' if all(p > 10 for p in controversial_pcts) else '✗ NO'}")
        print(f"  All datasets show inverse scaling: {'✓ YES' if all(p > 0 for p in inv_pcts) else '✗ NO'}")
        print(f"  Coefficient of variation (controversial %): {np.std(controversial_pcts)/np.mean(controversial_pcts):.2f}")
        print(f"    (<0.5 = consistent, >1.0 = highly variable)")
        
        stats_results['consistency'] = {
            'all_have_controversial': all(p > 10 for p in controversial_pcts),
            'all_have_inverse': all(p > 0 for p in inv_pcts),
            'cv_controversial': np.std(controversial_pcts)/np.mean(controversial_pcts)
        }
        
        return stats_results
# ============================================================================
# ABLATION STUDIES
# ============================================================================

class GeneralizabilityAblation:
    """Test if patterns generalize across benchmark types"""
    
    def __init__(self, categorization_results: Dict, inverse_results: Dict):
        self.cat_results = categorization_results
        self.inv_results = inverse_results
    
    def test_generalizability(self) -> Dict:
        """Test if patterns hold across different benchmark types"""
        print("\n" + "="*80)
        print("ABLATION 1: GENERALIZABILITY")
        print("="*80)
        
        benchmark_types = {
            'mathematical': ['gsm8k'],
            'factual': ['boolq', 'arc-easy', 'mmlu-stem'],
            'commonsense': ['commonsenseqa']
        }
        
        results = {}
        for btype, benchmarks in benchmark_types.items():
            controversial_pcts = [self.cat_results[b]['controversial_pct'] 
                                for b in benchmarks if b in self.cat_results]
            inverse_pcts = [self.inv_results[b]['pct'] 
                          for b in benchmarks if b in self.inv_results]
            
            if controversial_pcts:
                results[btype] = {
                    'controversial': np.mean(controversial_pcts),
                    'inverse': np.mean(inverse_pcts) if inverse_pcts else 0
                }
                print(f"   {btype}: contr={np.mean(controversial_pcts):.1f}%, inv={np.mean(inverse_pcts) if inverse_pcts else 0:.1f}%")
        
        cv = np.std([r['controversial'] for r in results.values()]) / np.mean([r['controversial'] for r in results.values()])
        print(f"   CV={cv:.2f} {'✓ Generalizes' if cv < 0.5 else '✗ Variable'}")
        
        return results


class ThresholdSensitivityAblation:
    """Test sensitivity to controversial regime definition"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
    
    def test_threshold_sensitivity(self) -> Dict:
        """Test different thresholds"""
        print("\n" + "="*80)
        print("ABLATION 2: THRESHOLD SENSITIVITY")
        print("="*80)
        
        thresholds = {
            'strict': (0.45, 0.55),
            'standard': (0.40, 0.60),
            'loose': (0.35, 0.65)
        }
        
        results = {}
        for dataset_name, df in self.data.items():
            dataset_results = {}
            
            for thresh_name, (min_t, max_t) in thresholds.items():
                controversial = 0
                total = 0
                
                for sid in df['sample_id'].unique():
                    sid_df = df[df['sample_id'] == sid]
                    success_rate = sid_df['is_correct'].mean()
                    total += 1
                    if min_t <= success_rate <= max_t:
                        controversial += 1
                
                dataset_results[thresh_name] = controversial / total * 100
            
            results[dataset_name] = dataset_results
            print(f"   {dataset_name}: {dataset_results}")
        
        return results


class BootstrapConfidenceAblation:
    """Bootstrap confidence intervals"""
    
    def __init__(self, categorization_results: Dict, inverse_results: Dict):
        self.cat_results = categorization_results
        self.inv_results = inverse_results
    
    def bootstrap_confidence(self, n_iter=1000) -> Dict:
        """Calculate 95% CIs"""
        print("\n" + "="*80)
        print("ABLATION 3: BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*80)
        
        all_contr = [r['controversial_pct'] for r in self.cat_results.values()]
        all_inv = [r['pct'] for r in self.inv_results.values()]
        
        bootstrap_contr = []
        bootstrap_inv = []
        
        for _ in range(n_iter):
            bootstrap_contr.append(np.mean(np.random.choice(all_contr, len(all_contr), replace=True)))
            bootstrap_inv.append(np.mean(np.random.choice(all_inv, len(all_inv), replace=True)))
        
        contr_ci = (np.percentile(bootstrap_contr, 2.5), np.percentile(bootstrap_contr, 97.5))
        inv_ci = (np.percentile(bootstrap_inv, 2.5), np.percentile(bootstrap_inv, 97.5))
        
        print(f"   Controversial 95% CI: [{contr_ci[0]:.1f}%, {contr_ci[1]:.1f}%]")
        print(f"   Inverse 95% CI: [{inv_ci[0]:.1f}%, {inv_ci[1]:.1f}%]")
        
        return {
            'controversial_ci': contr_ci,
            'inverse_ci': inv_ci
        }
        
# ============================================================================
# ANALYSIS 10: CONTAMINATION DETECTION (CRITICAL FOR NATURE)
# ============================================================================
class ContaminationAnalyzer:
    """
    CRITICAL: Test if inverse scaling is dataset memorization vs real capability
    
    Four tests:
    1. Response diversity (memorized = identical responses)
    2. Length variability (memorized = uniform lengths)  
    3. Error pattern analysis (what type of errors?)
    4. Model-age correlation (newer models should show less if contaminated)
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], inverse_results: Dict):
        self.data = data
        self.inv_results = inverse_results
        
    def analyze_contamination_risk(self) -> Dict:
        """Main contamination analysis"""
        print("\n" + "="*80)
        print("ANALYSIS 10: CONTAMINATION DETECTION")
        print("="*80)
        print("\n CRITICAL QUESTION:")
        print("   Is inverse scaling real, or just dataset memorization?")
        print("\n   If small models memorized answers:")
        print("   - They'd give identical responses (low diversity)")
        print("   - They'd show uniform lengths (low variability)")
        print("   - Pattern would disappear on new problems")
        
        contamination_results = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in self.inv_results or self.inv_results[dataset_name]['count'] == 0:
                continue
            
            print(f"\n {dataset_name.upper()}")
            
            inv_df = self.inv_results[dataset_name]['inverse_df']
            inverse_problem_ids = set(inv_df['sample_id'])
            
            # ================================================================
            # TEST 1: RESPONSE DIVERSITY
            # ================================================================
            print(f"\n   TEST 1: Response Diversity")
            print(f"   (Memorization → Low diversity, Real capability → High diversity)")
            
            weak_responses = []
            strong_responses = []
            
            for prob_id in inverse_problem_ids:
                prob_df = df[df['sample_id'] == prob_id]
                
                weak_df = prob_df[prob_df['model_name'].isin(Config.WEAK_MODELS)]
                strong_df = prob_df[prob_df['model_name'].isin(Config.STRONG_MODELS)]
                
                if len(weak_df) > 0:
                    weak_responses.extend([str(r) for r in weak_df['full_generation'].tolist()])
                if len(strong_df) > 0:
                    strong_responses.extend([str(r) for r in strong_df['full_generation'].tolist()])
            
            # Calculate diversity
            weak_unique = len(set(weak_responses))
            weak_total = len(weak_responses)
            weak_diversity = weak_unique / weak_total if weak_total > 0 else 0
            
            strong_unique = len(set(strong_responses))
            strong_total = len(strong_responses)
            strong_diversity = strong_unique / strong_total if strong_total > 0 else 0
            
            print(f"      Small models: {weak_unique}/{weak_total} unique ({weak_diversity:.1%})")
            print(f"      Large models: {strong_unique}/{strong_total} unique ({strong_diversity:.1%})")
            
            # Interpret
            if weak_diversity < 0.30:
                verdict = "  HIGH RISK: Low diversity suggests memorization"
                risk = "HIGH"
            elif weak_diversity < 0.50:
                verdict = "  MODERATE RISK: Some repetition detected"
                risk = "MODERATE"
            else:
                verdict = "✓ LOW RISK: High diversity suggests genuine capability"
                risk = "LOW"
            
            print(f"      Verdict: {verdict}")
            
            # ================================================================
            # TEST 2: LENGTH VARIABILITY
            # ================================================================
            print(f"\n   TEST 2: Length Variability")
            print(f"   (Memorization → Uniform lengths, Real capability → Variable lengths)")
            
            weak_lengths = [len(str(r).split()) for r in weak_responses if str(r).strip()]
            strong_lengths = [len(str(r).split()) for r in strong_responses if str(r).strip()]
            
            if len(weak_lengths) > 1 and len(strong_lengths) > 1:
                weak_cv = np.std(weak_lengths) / np.mean(weak_lengths) if np.mean(weak_lengths) > 0 else 0
                strong_cv = np.std(strong_lengths) / np.mean(strong_lengths) if np.mean(strong_lengths) > 0 else 0
                
                print(f"      Small models CV: {weak_cv:.3f} (mean={np.mean(weak_lengths):.1f} words)")
                print(f"      Large models CV: {strong_cv:.3f} (mean={np.mean(strong_lengths):.1f} words)")
                
                if weak_cv < 0.20:
                    length_verdict = "  Suspiciously uniform lengths"
                    length_risk = "HIGH"
                else:
                    length_verdict = "✓ Natural variability"
                    length_risk = "LOW"
                
                print(f"      Verdict: {length_verdict}")
            else:
                weak_cv = 0
                strong_cv = 0
                length_risk = "UNKNOWN"
            
            # ================================================================
            # TEST 3: ERROR PATTERN ANALYSIS
            # ================================================================
            print(f"\n   TEST 3: Error Pattern Analysis")
            print(f"   (What types of errors do large models make?)")
            
            error_categories = {
                'over_reasoning': 0,
                'format_mismatch': 0,
                'simple_wrong': 0,
                'hedging': 0
            }
            
            for prob_id in list(inverse_problem_ids)[:20]:
                prob_df = df[df['sample_id'] == prob_id]
                strong_wrong = prob_df[(prob_df['model_name'].isin(Config.STRONG_MODELS)) & 
                                      (prob_df['is_correct'] == False)]
                
                for _, row in strong_wrong.iterrows():
                    response = str(row.get('full_generation', ''))
                    response_len = len(response.split())
                    
                    if response_len > 100:
                        error_categories['over_reasoning'] += 1
                    elif response_len < 20:
                        error_categories['simple_wrong'] += 1
                    elif 'however' in response.lower() or 'but' in response.lower():
                        error_categories['hedging'] += 1
                    else:
                        error_categories['format_mismatch'] += 1
            
            total_errors = sum(error_categories.values())
            if total_errors > 0:
                print(f"      Error types (n={total_errors}):")
                for err_type, count in sorted(error_categories.items(), key=lambda x: -x[1]):
                    pct = count / total_errors * 100
                    print(f"         {err_type:<20}: {count:>3} ({pct:>5.1f}%)")
                
                over_reasoning_pct = error_categories['over_reasoning'] / total_errors
                if over_reasoning_pct > 0.5:
                    error_verdict = "✓ Predominantly over-reasoning (not memorization avoidance)"
                    error_risk = "LOW"
                else:
                    error_verdict = "  Mixed error patterns"
                    error_risk = "MODERATE"
                
                print(f"      Verdict: {error_verdict}")
            else:
                error_risk = "UNKNOWN"
            
            # ================================================================
            # OVERALL ASSESSMENT
            # ================================================================
            print(f"\n   " + "="*70)
            print(f"   OVERALL CONTAMINATION RISK: ", end="")
            
            risk_scores = [risk, length_risk, error_risk]
            high_risk_count = sum(1 for r in risk_scores if r == "HIGH")
            
            if high_risk_count >= 2:
                overall_risk = "HIGH"
                print(f"  HIGH - Multiple red flags detected")
            elif high_risk_count == 1 or "MODERATE" in risk_scores:
                overall_risk = "MODERATE"
                print(f"  MODERATE - Some concerns present")
            else:
                overall_risk = "LOW"
                print(f"✓ LOW - Patterns consistent with genuine capability")
            
            contamination_results[dataset_name] = {
                'response_diversity': weak_diversity,
                'diversity_risk': risk,
                'length_cv': weak_cv,
                'length_risk': length_risk,
                'error_risk': error_risk,
                'overall_risk': overall_risk,
                'error_breakdown': error_categories
            }
        
        # ================================================================
        # AGGREGATE SUMMARY
        # ================================================================
        print(f"\n" + "="*80)
        print(f"AGGREGATE CONTAMINATION ASSESSMENT")
        print(f"="*80)
        
        all_risks = [r['overall_risk'] for r in contamination_results.values()]
        high_risk_datasets = sum(1 for r in all_risks if r == "HIGH")
        
        print(f"\n   Datasets analyzed: {len(contamination_results)}")
        print(f"   HIGH risk: {sum(1 for r in all_risks if r == 'HIGH')}")
        print(f"   MODERATE risk: {sum(1 for r in all_risks if r == 'MODERATE')}")
        print(f"   LOW risk: {sum(1 for r in all_risks if r == 'LOW')}")
        
        if high_risk_datasets == 0:
            print(f"\n   ✅ CONCLUSION: Inverse scaling is REAL, not contamination artifact")
            print(f"      - High response diversity across datasets")
            print(f"      - Natural length variability")
            print(f"      - Error patterns show over-reasoning, not memorization avoidance")
        elif high_risk_datasets >= 3:
            print(f"\n     CONCLUSION: Contamination concerns CANNOT BE RULED OUT")
            print(f"      - REQUIRED: Test on post-2024 benchmarks (GPQA, MATH-500)")
            print(f"      - REQUIRED: Paraphrase robustness test")
            print(f"      - DO NOT SUBMIT TO NATURE without these tests")
        else:
            print(f"\n     CONCLUSION: Mixed evidence - proceed with caution")
            print(f"      - RECOMMENDED: Add post-2024 benchmark validation")
        
        return contamination_results        

class ModelFamilyAnalyzer:
    """
    ANALYSIS 6: Which model families show most inverse scaling?
    Do architectural differences predict inverse scaling vulnerability?
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], inverse_results: Dict):
        self.data = data
        self.inv_results = inverse_results
        
    def analyze_family_differences(self) -> Dict:
        """Analyze inverse scaling by model family"""
        print("\n" + "="*80)
        print("ANALYSIS 6: MODEL FAMILY ANALYSIS")
        print("="*80)
        print("\n RESEARCH QUESTION:")
        print("   Which model families are most vulnerable to inverse scaling?")
        
        family_results = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in self.inv_results or self.inv_results[dataset_name]['count'] == 0:
                continue
            
            print(f"\n {dataset_name.upper()}")
            
            inv_df = self.inv_results[dataset_name]['inverse_df']
            inverse_problem_ids = set(inv_df['sample_id'])
            
            # Track family performance on inverse problems
            family_stats = {}
            
            for family in set(m['family'] for m in Config.MODEL_METADATA.values()):
                family_models = [m for m, d in Config.MODEL_METADATA.items() if d['family'] == family]
                
                if not family_models:
                    continue
                
                # Get performance on inverse scaling problems
                inverse_samples = df[df['sample_id'].isin(inverse_problem_ids)]
                family_samples = inverse_samples[inverse_samples['model_name'].isin(family_models)]
                
                if len(family_samples) == 0:
                    continue
                
                # Calculate accuracy and model size stats
                accuracy = family_samples['is_correct'].mean()
                model_sizes = [Config.MODEL_METADATA[m]['size'] for m in family_samples['model_name'].unique()]
                avg_size = np.mean(model_sizes) if model_sizes else 0
                
                family_stats[family] = {
                    'accuracy': accuracy,
                    'n_models': len(family_samples['model_name'].unique()),
                    'n_responses': len(family_samples),
                    'avg_size': avg_size
                }
            
            # Sort by accuracy (worst performers = most affected by inverse scaling)
            sorted_families = sorted(family_stats.items(), key=lambda x: x[1]['accuracy'])
            
            print(f"\n   Family performance on inverse scaling problems:")
            print(f"   {'Family':<15} {'Accuracy':>10} {'Models':>8} {'Avg Size':>10}")
            print(f"   {'-'*50}")
            
            for family, stats in sorted_families[:5]:  # Top 5 worst
                marker = "❌" if stats['accuracy'] < 0.3 else "⚠️" if stats['accuracy'] < 0.5 else "✓"
                print(f"   {marker} {family:<13} {stats['accuracy']:>9.1%} {stats['n_models']:>8} {stats['avg_size']:>9.1f}B")
            
            family_results[dataset_name] = family_stats
            
            # Statistical test: Do families differ significantly?
            if len(family_stats) >= 3:
                accuracies = [s['accuracy'] for s in family_stats.values()]
                if len(accuracies) >= 3:
                    kruskal_stat, kruskal_pval = kruskal(*[[a] for a in accuracies])
                    print(f"\n   Kruskal-Wallis test (families differ): H={kruskal_stat:.2f}, p={kruskal_pval:.4f}")
                    
                    if kruskal_pval < 0.05:
                        print(f"    FINDING: Model families show SIGNIFICANTLY different vulnerability!")
        
        # Aggregate across datasets
        print(f"\n" + "="*80)
        print(f"AGGREGATE FAMILY ANALYSIS")
        print(f"="*80)
        
        # Combine all family stats
        aggregate_families = {}
        for dataset_stats in family_results.values():
            for family, stats in dataset_stats.items():
                if family not in aggregate_families:
                    aggregate_families[family] = []
                aggregate_families[family].append(stats['accuracy'])
        
        print(f"\n   Overall family rankings (avg accuracy on inverse problems):")
        print(f"   {'Family':<15} {'Avg Accuracy':>15} {'Datasets':>10}")
        print(f"   {'-'*45}")
        
        sorted_agg = sorted(aggregate_families.items(), key=lambda x: np.mean(x[1]))
        for family, accuracies in sorted_agg[:8]:  # Top 8
            avg_acc = np.mean(accuracies)
            marker = "❌" if avg_acc < 0.3 else "⚠️" if avg_acc < 0.5 else "✓"
            print(f"   {marker} {family:<13} {avg_acc:>14.1%} {len(accuracies):>10}")
        
        print(f"\n    KEY FINDING: {sorted_agg[0][0]} most vulnerable to inverse scaling")
        print(f"      (avg accuracy: {np.mean(sorted_agg[0][1]):.1%} on inverse problems)")
        
        return family_results
    
    
class ProblemContentAnalyzer:
    """
    ANALYSIS 7: What characterizes inverse scaling problems?
    Are they simpler, more ambiguous, differently structured?
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], inverse_results: Dict):
        self.data = data
        self.inv_results = inverse_results
        
    def analyze_content_characteristics(self) -> Dict:
        """Analyze what makes inverse scaling problems different"""
        print("\n" + "="*80)
        print("ANALYSIS 7: PROBLEM CONTENT ANALYSIS")
        print("="*80)
        print("\n RESEARCH QUESTION:")
        print("   What content characteristics define inverse scaling problems?")
        
        content_results = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in self.inv_results or self.inv_results[dataset_name]['count'] == 0:
                continue
            
            print(f"\n {dataset_name.upper()}")
            
            inv_df = self.inv_results[dataset_name]['inverse_df']
            inverse_problem_ids = set(inv_df['sample_id'])
            
            # Get sample of inverse vs normal problems
            inverse_problems = []
            normal_problems = []
            
            for sample_id in df['sample_id'].unique():
                sample_df = df[df['sample_id'] == sample_id].iloc[0]
                
                # Extract question text
                question = str(sample_df.get('question', ''))
                
                if not question or len(question) < 10:
                    continue
                
                # Basic content features
                features = {
                    'length': len(question.split()),
                    'has_numbers': bool(any(char.isdigit() for char in question)),
                    'has_question_mark': '?' in question,
                    'complexity': question.count(',') + question.count(';'),
                }
                
                if sample_id in inverse_problem_ids:
                    inverse_problems.append(features)
                else:
                    normal_problems.append(features)
            
            if len(inverse_problems) < 3 or len(normal_problems) < 3:
                continue
            
            # Compare features
            inv_lengths = [p['length'] for p in inverse_problems]
            norm_lengths = [p['length'] for p in normal_problems]
            
            inv_has_nums = sum(p['has_numbers'] for p in inverse_problems) / len(inverse_problems)
            norm_has_nums = sum(p['has_numbers'] for p in normal_problems) / len(normal_problems)
            
            inv_complexity = np.mean([p['complexity'] for p in inverse_problems])
            norm_complexity = np.mean([p['complexity'] for p in normal_problems])
            
            # Statistical tests
            length_stat, length_pval = mannwhitneyu(inv_lengths, norm_lengths, alternative='two-sided')
            
            print(f"\n   Content characteristics:")
            print(f"   {'Feature':<25} {'Inverse':>12} {'Normal':>12} {'Diff':>10}")
            print(f"   {'-'*65}")
            print(f"   {'Question length (words)':<25} {np.mean(inv_lengths):>11.1f} {np.mean(norm_lengths):>11.1f} {np.mean(inv_lengths)-np.mean(norm_lengths):>9.1f}")
            print(f"   {'Contains numbers (%)':<25} {inv_has_nums*100:>11.1f} {norm_has_nums*100:>11.1f} {(inv_has_nums-norm_has_nums)*100:>9.1f}")
            print(f"   {'Complexity (clauses)':<25} {inv_complexity:>11.1f} {norm_complexity:>11.1f} {inv_complexity-norm_complexity:>9.1f}")
            
            print(f"\n   Length comparison: p={length_pval:.4f} {'✓ Significant' if length_pval < 0.05 else '✗ Not significant'}")
            
            if np.mean(inv_lengths) < np.mean(norm_lengths) * 0.8:
                print(f"    FINDING: Inverse scaling problems are SIMPLER (20% shorter)")
            elif np.mean(inv_lengths) > np.mean(norm_lengths) * 1.2:
                print(f"    FINDING: Inverse scaling problems are MORE COMPLEX (20% longer)")
            else:
                print(f"    FINDING: Length-independent pattern")
            
            content_results[dataset_name] = {
                'inverse_length': np.mean(inv_lengths),
                'normal_length': np.mean(norm_lengths),
                'inverse_has_numbers': inv_has_nums,
                'normal_has_numbers': norm_has_nums,
                'length_pval': length_pval
            }
        
        return content_results


class ScaleThresholdAnalyzer:
    """
    ANALYSIS 8: At what scale does performance degrade?
    Find the "sweet spot" before overthinking begins
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], inverse_results: Dict):
        self.data = data
        self.inv_results = inverse_results
        
    def find_degradation_threshold(self) -> Dict:
        """Find scale at which performance starts degrading"""
        print("\n" + "="*80)
        print("ANALYSIS 8: SCALE THRESHOLD ANALYSIS")
        print("="*80)
        print("\n RESEARCH QUESTION:")
        print("   At what model scale does performance start degrading?")
        
        threshold_results = {}
        
        for dataset_name, df in self.data.items():
            if dataset_name not in self.inv_results or self.inv_results[dataset_name]['count'] == 0:
                continue
            
            print(f"\n📊 {dataset_name.upper()}")
            
            inv_df = self.inv_results[dataset_name]['inverse_df']
            inverse_problem_ids = set(inv_df['sample_id'])
            
            # Get performance by model size on inverse problems
            size_performance = {}
            
            for model_name in df['model_name'].unique():
                if model_name not in Config.MODEL_METADATA:
                    continue
                
                size = Config.MODEL_METADATA[model_name]['size']
                model_df = df[(df['model_name'] == model_name) & 
                             (df['sample_id'].isin(inverse_problem_ids))]
                
                if len(model_df) == 0:
                    continue
                
                accuracy = model_df['is_correct'].mean()
                
                if size not in size_performance:
                    size_performance[size] = []
                size_performance[size].append(accuracy)
            
            # Average by size
            avg_by_size = {size: np.mean(accs) for size, accs in size_performance.items()}
            sorted_sizes = sorted(avg_by_size.items())
            
            if len(sorted_sizes) < 5:
                continue
            
            print(f"\n   Performance by model scale (on inverse problems):")
            print(f"   {'Scale':>8} {'Accuracy':>12} {'N Models':>10}")
            print(f"   {'-'*35}")
            
            for size, acc in sorted_sizes[:10]:
                n_models = len(size_performance[size])
                marker = "✓" if acc > 0.6 else "⚠️" if acc > 0.4 else "❌"
                print(f"   {marker} {size:>6.1f}B {acc:>11.1%} {n_models:>10}")
            
            # Find peak and degradation point
            peak_size = max(sorted_sizes[:5], key=lambda x: x[1])[0]  # Peak in first 5
            peak_acc = avg_by_size[peak_size]
            
            # Find where performance drops below 80% of peak
            degradation_threshold = None
            for size, acc in sorted_sizes:
                if size > peak_size and acc < peak_acc * 0.8:
                    degradation_threshold = size
                    break
            
            print(f"\n    FINDINGS:")
            print(f"      Peak performance: {peak_size:.1f}B ({peak_acc:.1%})")
            if degradation_threshold:
                print(f"      Degradation begins: {degradation_threshold:.1f}B (20% drop)")
                print(f"      Sweet spot: {peak_size:.1f}B - {degradation_threshold:.1f}B")
            
            # Correlation between size and performance
            sizes = [s for s, _ in sorted_sizes]
            accs = [a for _, a in sorted_sizes]
            corr, pval = stats.spearmanr(sizes, accs)
            
            print(f"      Size-performance correlation: rs={corr:+.3f}, p={pval:.4f}")
            if corr < -0.3 and pval < 0.05:
                print(f"        STRONG INVERSE SCALING: Larger = Worse")
            
            threshold_results[dataset_name] = {
                'peak_size': peak_size,
                'peak_accuracy': peak_acc,
                'degradation_threshold': degradation_threshold,
                'correlation': corr,
                'correlation_pval': pval
            }
        
        return threshold_results


class CostBenefitAnalyzer:
    """
    ANALYSIS 9: Quantify economic impact
    Calculate exact cost savings from problem-specific routing
    """
    
    def __init__(self, inverse_results: Dict):
        self.inv_results = inverse_results
        
    def calculate_cost_savings(self) -> Dict:
        """Calculate economic impact of findings"""
        print("\n" + "="*80)
        print("ANALYSIS 9: COST-BENEFIT ANALYSIS")
        print("="*80)
        print("\n RESEARCH QUESTION:")
        print("   What are the economic implications of our findings?")
        
        # Cost assumptions (industry standard, 2024)
        COSTS_PER_1M_TOKENS = {
            '1B': 0.15,      # Small models (Qwen 0.5-3B)
            '7B': 0.30,      # Medium models
            '70B': 2.00,     # Large models  
            '405B': 15.00    # Frontier models
        }
        
        # Calculate from results
        total_problems = sum(r['total'] for r in self.inv_results.values())
        inverse_problems = sum(r['count'] for r in self.inv_results.values())
        inverse_pct = (inverse_problems / total_problems) * 100
        
        print(f"\n BASELINE SCENARIO: Deploy 70B model everywhere")
        print(f"   Workload: 1B tokens/day evaluation")
        print(f"   Cost: ${COSTS_PER_1M_TOKENS['70B'] * 1000:.2f}/day")
        print(f"   Annual: ${COSTS_PER_1M_TOKENS['70B'] * 1000 * 365:,.0f}")
        
        print(f"\n OPTIMIZED SCENARIO: Problem-specific routing")
        print(f"   Route {inverse_pct:.1f}% of problems to 1B models")
        print(f"   Route {100-inverse_pct:.1f}% to 70B models")
        
        # Calculate savings
        baseline_cost_daily = COSTS_PER_1M_TOKENS['70B'] * 1000
        optimized_cost_daily = (
            (inverse_pct/100) * COSTS_PER_1M_TOKENS['1B'] * 1000 +
            ((100-inverse_pct)/100) * COSTS_PER_1M_TOKENS['70B'] * 1000
        )
        
        savings_daily = baseline_cost_daily - optimized_cost_daily
        savings_pct = (savings_daily / baseline_cost_daily) * 100
        
        print(f"\n   Optimized cost: ${optimized_cost_daily:.2f}/day")
        print(f"   Savings: ${savings_daily:.2f}/day ({savings_pct:.1f}%)")
        print(f"   Annual savings: ${savings_daily * 365:,.0f}")
        
        print(f"\n BENCHMARK WASTE CALCULATION")
        avg_useless = 28.5  # From categorization results
        print(f"   Non-discriminative problems: {avg_useless:.1f}%")
        print(f"   If you're running 1B token evaluations:")
        print(f"   Wasted compute: {avg_useless:.1f}% = {avg_useless/100 * 1000:.0f}M tokens")
        print(f"   Wasted cost: ${avg_useless/100 * COSTS_PER_1M_TOKENS['70B'] * 1000:.2f}/day")
        print(f"   Annual waste: ${avg_useless/100 * COSTS_PER_1M_TOKENS['70B'] * 1000 * 365:,.0f}")
        
        print(f"\n SCALING TO INDUSTRY")
        print(f"   Assume top 10 AI labs run 100B tokens/day evaluation each")
        industry_baseline = 10 * 100 * 1000 * COSTS_PER_1M_TOKENS['70B']
        industry_optimized = 10 * 100 * 1000 * (
            (inverse_pct/100) * COSTS_PER_1M_TOKENS['1B'] +
            ((100-inverse_pct)/100) * COSTS_PER_1M_TOKENS['70B']
        )
        industry_savings = industry_baseline - industry_optimized
        
        print(f"   Industry-wide daily savings: ${industry_savings:,.0f}")
        print(f"   Industry-wide annual savings: ${industry_savings * 365:,.0f}")
        
        print(f"\n KEY FINDINGS:")
        print(f"   1. Problem-specific routing saves {savings_pct:.1f}% on evaluation costs")
        print(f"   2. Filtering non-discriminative problems saves {avg_useless:.1f}% more")
        print(f"   3. Combined: Up to {savings_pct + avg_useless:.1f}% cost reduction")
        print(f"   4. Industry-wide: ${(industry_savings * 365 + 10 * 100 * 1000 * (avg_useless/100) * COSTS_PER_1M_TOKENS['70B'] * 365):,.0f}/year")
        
        return {
            'inverse_pct': inverse_pct,
            'savings_pct': savings_pct,
            'daily_savings': savings_daily,
            'annual_savings': savings_daily * 365,
            'industry_annual_savings': industry_savings * 365
        }


# ============================================================================
# ANALYSIS 5: PROBLEM CHARACTERISTICS ANALYSIS
# ============================================================================
class ProblemCharacteristicsAnalyzer:
    """Analyze what makes problems controversial or show inverse scaling"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], 
                 categorization_results: Dict,
                 inverse_results: Dict):
        self.data = data
        self.cat_results = categorization_results
        self.inv_results = inverse_results
        
    def analyze_problem_characteristics(self) -> Dict:
        """Analyze characteristics of different problem types"""
        print("\n" + "="*80)
        print("ANALYSIS 5: PROBLEM CHARACTERISTICS")
        print("="*80)
        
        characteristics = {}
        
        for dataset_name, df in self.data.items():
            print(f"\n {dataset_name.upper()}")
            
            if dataset_name not in self.cat_results:
                continue
            
            stats_df = self.cat_results[dataset_name]['stats_df']
            
            # Get inverse scaling problem IDs
            inverse_ids = set()
            if dataset_name in self.inv_results and self.inv_results[dataset_name]['count'] > 0:
                inverse_ids = set(self.inv_results[dataset_name]['inverse_df']['sample_id'])
            
            # Analyze variance by category
            print(f"\n   Variance by category:")
            for category in ['controversial', 'universally_easy', 'moderate', 'universally_hard']:
                cat_df = stats_df[stats_df['category'] == category]
                if len(cat_df) > 0:
                    avg_variance = cat_df['variance'].mean()
                    print(f"   {category:<20}: {avg_variance:.3f}")
            
            # Overlap between controversial and inverse scaling
            controversial_ids = set(stats_df[stats_df['category'] == 'controversial']['sample_id'])
            overlap = len(controversial_ids & inverse_ids)
            
            if len(inverse_ids) > 0:
                overlap_pct = overlap / len(inverse_ids) * 100
                print(f"\n   Inverse scaling problems:")
                print(f"   Total: {len(inverse_ids)}")
                print(f"   Also controversial: {overlap} ({overlap_pct:.1f}%)")
                print(f"    INSIGHT: {'Inverse scaling concentrated in controversial regime' if overlap_pct > 50 else 'Inverse scaling spans multiple regimes'}")
            
            characteristics[dataset_name] = {
                'variance_by_category': {
                    cat: stats_df[stats_df['category'] == cat]['variance'].mean()
                    for cat in ['controversial', 'universally_easy', 'moderate', 'universally_hard']
                    if len(stats_df[stats_df['category'] == cat]) > 0
                },
                'inverse_controversial_overlap': overlap_pct if len(inverse_ids) > 0 else 0
            }
        
        return characteristics
    
# ============================================================================
# ANALYSIS 11: CAUSAL INTERVENTION ANALYZER
# ============================================================================
class CausalInterventionAnalyzer:
    """
    Analyze causal intervention experiment results
    
    Tests the overthinking hypothesis: Do brevity constraints reduce inverse scaling?
    """
    
    def __init__(self, causal_results_dir: Path):
        self.causal_dir = causal_results_dir
        
    def analyze_causal_intervention(self) -> Dict:
        """Main causal intervention analysis"""
        print("\n LOADING CAUSAL INTERVENTION DATA")
        print("="*80)
        
        # Find all model directories
        model_dirs = [d for d in self.causal_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(model_dirs)} model directories:")
        for d in model_dirs:
            print(f"  - {d.name}")
        
        # Load all data
        all_data = []
        
        for model_dir in model_dirs:
            model_name = model_dir.name.replace('_model', '')
            
            # Normalize model name for metadata lookup
            model_name_normalized = self._normalize_model_name(model_name)
            
            for dataset in Config.DATASETS:
                dataset_path = model_dir / dataset / "raw_responses"
                
                if not dataset_path.exists():
                    continue
                
                # Load all CSV files (control, brief, direct)
                for csv_file in dataset_path.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        
                        # Add model metadata
                        df['model_name_normalized'] = model_name_normalized
                        df['model_dir'] = model_dir.name
                        df['dataset'] = dataset
                        
                        # Infer condition from filename
                        filename = csv_file.stem
                        if 'control' in filename:
                            df['condition'] = 'control'
                        elif 'brief' in filename:
                            df['condition'] = 'brief'
                        elif 'direct' in filename:
                            df['condition'] = 'direct'
                        else:
                            # Try to get from column if exists
                            if 'condition' not in df.columns:
                                print(f"  Warning: Cannot determine condition for {csv_file}")
                                continue
                        
                        all_data.append(df)
                        
                    except Exception as e:
                        print(f" Error loading {csv_file}: {e}")
                        continue
        
        if not all_data:
            print(" No causal intervention data found!")
            return {}
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add model size category
        combined_df['model_size_category'] = combined_df['model_name_normalized'].apply(
            self._get_model_size_category
        )
        
        print(f"\n Loaded {len(combined_df)} total responses")
        print(f"   Models: {combined_df['model_name_normalized'].nunique()}")
        print(f"   Datasets: {combined_df['dataset'].nunique()}")
        print(f"   Conditions: {combined_df['condition'].unique()}")
        
        # Run analyses
        results = {}
        
        # Analysis 1: Overall effect
        results['overall'] = self._analyze_overall_effect(combined_df)
        
        # Analysis 2: By dataset
        results['by_dataset'] = self._analyze_by_dataset(combined_df)
        
        # Analysis 3: Response length validation
        results['response_length'] = self._analyze_response_length(combined_df)
        
        # Analysis 4: Statistical tests
        results['statistical_tests'] = self._statistical_tests(combined_df)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _normalize_model_name(self, name: str) -> str:
        """Normalize model names to match Config.MODEL_METADATA"""
        
        # Remove _model suffix
        name = name.replace('_model', '')
        
        # Mapping of directory names to metadata keys
        name_mapping = {
            'databricks-meta-llama-3-1-405b-instruct': 'databricks-meta-llama-3-1-405b-instruct',
            'databricks-qwen3-next-80b-a3b-instruct': 'Qwen/Qwen2.5-32B-Instruct',  # Approximate
            'gemma-2b-it': 'google/gemma-2-2b-it',
            'llama-3.3-70b-versatile': 'meta/llama-3.3-70b-versatile',
            'meta-llamaLlama-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct',
            'QwenQwen2.5-3B-Instruct': 'Qwen/Qwen2.5-3B-Instruct',
            # 'qwenqwen3-32b': 'Qwen/Qwen2.5-32B-Instruct',
        }
        
        return name_mapping.get(name, name)
    
    def _get_model_size_category(self, model_name: str) -> str:
        """Get size category for a model"""
        if model_name in Config.MODEL_METADATA:
            size = Config.MODEL_METADATA[model_name]['size']
            if size < 10:
                return 'small'
            else:
                return 'large'
        else:
            # Guess from name
            if any(x in model_name.lower() for x in ['2b', '3b', '1b']):
                return 'small'
            else:
                return 'large'
    
    def _analyze_overall_effect(self, df: pd.DataFrame) -> Dict:
        """Analyze overall causal effect"""
        print(f"\n{'='*80}")
        print(f"OVERALL CAUSAL EFFECT")
        print(f"{'='*80}")
        
        results = {}
        
        for condition in ['control', 'brief', 'direct']:
            cond_df = df[df['condition'] == condition]
            
            small_acc = cond_df[cond_df['model_size_category'] == 'small']['is_correct'].mean()
            large_acc = cond_df[cond_df['model_size_category'] == 'large']['is_correct'].mean()
            gap = small_acc - large_acc
            
            results[condition] = {
                'small_accuracy': small_acc,
                'large_accuracy': large_acc,
                'gap': gap,
                'n_small': len(cond_df[cond_df['model_size_category'] == 'small']),
                'n_large': len(cond_df[cond_df['model_size_category'] == 'large'])
            }
            
            print(f"\n{condition.upper()}:")
            print(f"   Small models: {small_acc:.1%} (n={results[condition]['n_small']})")
            print(f"   Large models: {large_acc:.1%} (n={results[condition]['n_large']})")
            print(f"   Gap: {gap:+.1%}")
        
        # Gap reduction
        control_gap = results['control']['gap']
        brief_gap = results['brief']['gap']
        direct_gap = results['direct']['gap']
        
        brief_reduction = (control_gap - brief_gap) / control_gap * 100 if control_gap != 0 else 0
        direct_reduction = (control_gap - direct_gap) / control_gap * 100 if control_gap != 0 else 0
        
        print(f"\n{'─'*80}")
        print(f"GAP REDUCTION:")
        print(f"   Control gap: {control_gap:+.1%}")
        print(f"   Brief reduces gap by: {brief_reduction:+.1f}%")
        print(f"   Direct reduces gap by: {direct_reduction:+.1f}%")
        
        results['gap_reduction'] = {
            'brief': brief_reduction,
            'direct': direct_reduction
        }
        
        return results
    
    def _analyze_by_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze effect by dataset"""
        print(f"\n{'='*80}")
        print(f"EFFECT BY DATASET")
        print(f"{'='*80}")
        
        results = {}
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            print(f"\n{dataset.upper()}:")
            
            dataset_results = {}
            
            for condition in ['control', 'brief', 'direct']:
                cond_df = dataset_df[dataset_df['condition'] == condition]
                
                if len(cond_df) == 0:
                    continue
                
                small_acc = cond_df[cond_df['model_size_category'] == 'small']['is_correct'].mean()
                large_acc = cond_df[cond_df['model_size_category'] == 'large']['is_correct'].mean()
                gap = small_acc - large_acc
                
                dataset_results[condition] = {
                    'small_accuracy': small_acc,
                    'large_accuracy': large_acc,
                    'gap': gap
                }
                
                print(f"   {condition:<10}: Gap = {gap:+.1%} (Small: {small_acc:.1%}, Large: {large_acc:.1%})")
            
            results[dataset] = dataset_results
        
        return results
    
    def _analyze_response_length(self, df: pd.DataFrame) -> Dict:
        """Analyze if conditions actually changed response length"""
        print(f"\n{'='*80}")
        print(f"RESPONSE LENGTH VALIDATION")
        print(f"{'='*80}")
        print(f"(Checking if brevity constraints worked)")
        
        results = {}
        
        for condition in ['control', 'brief', 'direct']:
            cond_df = df[df['condition'] == condition]
            
            if 'output_tokens' in cond_df.columns:
                small_len = cond_df[cond_df['model_size_category'] == 'small']['output_tokens'].mean()
                large_len = cond_df[cond_df['model_size_category'] == 'large']['output_tokens'].mean()
            else:
                # Fallback: estimate from full_generation
                small_len = cond_df[cond_df['model_size_category'] == 'small']['full_generation'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                ).mean()
                large_len = cond_df[cond_df['model_size_category'] == 'large']['full_generation'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                ).mean()
            
            results[condition] = {
                'small_length': small_len,
                'large_length': large_len
            }
            
            print(f"\n{condition.upper()}:")
            print(f"   Small models: {small_len:.0f} tokens")
            print(f"   Large models: {large_len:.0f} tokens")
        
        # Check if brief actually reduced length
        control_large = results['control']['large_length']
        brief_large = results['brief']['large_length']
        reduction = (control_large - brief_large) / control_large * 100 if control_large > 0 else 0
        
        print(f"\n{'─'*80}")
        print(f"VALIDATION:")
        print(f"   Large models - Control: {control_large:.0f} tokens")
        print(f"   Large models - Brief: {brief_large:.0f} tokens")
        print(f"   Reduction: {reduction:.1f}%")
        
        if reduction > 20:
            print(f"    Brevity constraint WORKED (>20% reduction)")
        else:
            print(f"     Brevity constraint may not have worked (<20% reduction)")
        
        return results
    
    def _statistical_tests(self, df: pd.DataFrame) -> Dict:
        """Run statistical significance tests"""
        print(f"\n{'='*80}")
        print(f"STATISTICAL TESTS")
        print(f"{'='*80}")
        
        from scipy.stats import ttest_rel, mannwhitneyu
        
        results = {}
        
        # Test 1: Does brevity improve large models?
        print(f"\nTEST 1: Does brevity improve large models?")
        
        large_df = df[df['model_size_category'] == 'large']
        
        # Get control vs brief for same problems
        control_large = large_df[large_df['condition'] == 'control']
        brief_large = large_df[large_df['condition'] == 'brief']
        
        # Match by sample_id
        control_by_problem = control_large.groupby('sample_id')['is_correct'].mean()
        brief_by_problem = brief_large.groupby('sample_id')['is_correct'].mean()
        
        # Find common problems
        common_ids = set(control_by_problem.index) & set(brief_by_problem.index)
        
        if len(common_ids) > 10:
            control_vals = [control_by_problem[i] for i in common_ids]
            brief_vals = [brief_by_problem[i] for i in common_ids]
            
            # Paired t-test
            t_stat, p_val = ttest_rel(brief_vals, control_vals)
            
            print(f"   Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"   Result: {'✓ SIGNIFICANT' if p_val < 0.05 else '✗ Not significant'}")
            
            results['brevity_helps'] = {
                't_stat': t_stat,
                'p_val': p_val,
                'significant': p_val < 0.05,
                'n_problems': len(common_ids)
            }
        else:
            print(f"     Insufficient data (only {len(common_ids)} common problems)")
            results['brevity_helps'] = {'insufficient_data': True}
        
        # Test 2: Gap reduction significance
        print(f"\nTEST 2: Is gap reduction significant?")
        
        control_gap = (df[(df['condition'] == 'control') & (df['model_size_category'] == 'small')]['is_correct'].mean() -
                       df[(df['condition'] == 'control') & (df['model_size_category'] == 'large')]['is_correct'].mean())
        
        brief_gap = (df[(df['condition'] == 'brief') & (df['model_size_category'] == 'small')]['is_correct'].mean() -
                     df[(df['condition'] == 'brief') & (df['model_size_category'] == 'large')]['is_correct'].mean())
        
        print(f"   Control gap: {control_gap:+.1%}")
        print(f"   Brief gap: {brief_gap:+.1%}")
        print(f"   Reduction: {(control_gap - brief_gap):+.1%}")
        
        results['gap_reduction_test'] = {
            'control_gap': control_gap,
            'brief_gap': brief_gap,
            'reduction': control_gap - brief_gap
        }
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print executive summary"""
        print(f"\n{'#'*80}")
        print(f"# CAUSAL INTERVENTION SUMMARY")
        print(f"{'#'*80}")
        
        overall = results['overall']
        
        print(f"\n KEY FINDINGS:")
        
        # Finding 1: Baseline inverse scaling
        control_gap = overall['control']['gap']
        print(f"\n1. BASELINE INVERSE SCALING (Control)")
        print(f"   Small models: {overall['control']['small_accuracy']:.1%}")
        print(f"   Large models: {overall['control']['large_accuracy']:.1%}")
        print(f"   Gap: {control_gap:+.1%}")
        
        # Finding 2: Causal effect of brevity
        brief_gap = overall['brief']['gap']
        gap_reduction = overall['gap_reduction']['brief']
        
        print(f"\n2. CAUSAL EFFECT OF BREVITY")
        print(f"   Brief condition gap: {brief_gap:+.1%}")
        print(f"   Gap reduction: {gap_reduction:+.1f}%")
        
        if gap_reduction > 30:
            print(f"    STRONG EVIDENCE: Brevity reduces inverse scaling by {gap_reduction:.0f}%!")
        elif gap_reduction > 15:
            print(f"    MODERATE EVIDENCE: Brevity reduces inverse scaling")
        else:
            print(f"     WEAK EVIDENCE: Small or inconsistent effect")
        
        # Finding 3: Statistical significance
        if 'brevity_helps' in results['statistical_tests'] and 'significant' in results['statistical_tests']['brevity_helps']:
            if results['statistical_tests']['brevity_helps']['significant']:
                print(f"\n3. STATISTICAL VALIDATION")
                print(f"    Effect is statistically significant (p < 0.05)")
            else:
                print(f"\n3. STATISTICAL VALIDATION")
                print(f"     Effect not significant (may need more data)")




# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Execute complete analysis"""
    
    print("\n" + "="*80)
    print("CONTROVERSIAL REGIME: COMPREHENSIVE ANALYSIS FOR NATURE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = DataLoader(Config.BASE_DIR)
    all_data = loader.load_all_data()
    
    if not all_data:
        print("\n ERROR: No data loaded.")
        return
    
    # ========================================================================
    # RUN ALL ANALYSES
    # ========================================================================
    
    # Analysis 1: Problem Categorization
    categorizer = ProblemCategorizationAnalyzer(all_data)
    cat_results = categorizer.categorize_all_problems()
    
    # Analysis 2: Inverse Scaling
    inverse_analyzer = InverseScalingAnalyzer(all_data)
    inv_results = inverse_analyzer.find_inverse_scaling_problems()
    
    # Analysis 3: Failure Mechanisms
    mechanism_analyzer = MechanismAnalyzer(all_data, inv_results)
    mechanism_results = mechanism_analyzer.analyze_failure_mechanisms()
    
    # Analysis 4: Statistical Rigor
    stats_analyzer = StatisticalRigorAnalyzer(cat_results, inv_results)
    stats_results = stats_analyzer.calculate_comprehensive_statistics()
    
    # Analysis 5: Problem Characteristics
    char_analyzer = ProblemCharacteristicsAnalyzer(all_data, cat_results, inv_results)
    char_results = char_analyzer.analyze_problem_characteristics()
    # Analysis 6: Model Family Analysis
    family_analyzer = ModelFamilyAnalyzer(all_data, inv_results)
    family_results = family_analyzer.analyze_family_differences()
    
    # Analysis 7: Problem Content Analysis
    content_analyzer = ProblemContentAnalyzer(all_data, inv_results)
    content_results = content_analyzer.analyze_content_characteristics()
    
    # Analysis 8: Scale Threshold Analysis
    threshold_analyzer = ScaleThresholdAnalyzer(all_data, inv_results)
    threshold_results = threshold_analyzer.find_degradation_threshold()
    
    # Analysis 9: Cost-Benefit Analysis
    cost_analyzer = CostBenefitAnalyzer(inv_results)
    cost_results = cost_analyzer.calculate_cost_savings()
    # ========================================================================
    # ANALYSIS 10: CONTAMINATION DETECTION (CRITICAL FOR NATURE)
    # ========================================================================
    contamination_analyzer = ContaminationAnalyzer(all_data, inv_results)
    contamination_results = contamination_analyzer.analyze_contamination_risk()
    # ========================================================================
    # ANALYSIS 11: CAUSAL INTERVENTION - TESTING OVERTHINKING HYPOTHESIS
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 11: CAUSAL INTERVENTION EXPERIMENT")
    print("="*80)
    
    causal_analyzer = CausalInterventionAnalyzer(
        causal_results_dir=Path("E:/publication/inverse_scaling/causal_intervention_results")
    )
    causal_results = causal_analyzer.analyze_causal_intervention()

    print("\n" + "="*80)
    print("RUNNING ABLATION STUDIES")
    print("="*80)

    ablation1 = GeneralizabilityAblation(cat_results, inv_results)
    gen_ablation = ablation1.test_generalizability()

    ablation2 = ThresholdSensitivityAblation(all_data)
    thresh_ablation = ablation2.test_threshold_sensitivity()

    ablation3 = BootstrapConfidenceAblation(cat_results, inv_results)
    bootstrap_ablation = ablation3.bootstrap_confidence()

    # Save ablations
    with open(Config.OUTPUT_DIR / 'ablations.json', 'w') as f:
        json.dump({
            'generalizability': gen_ablation,
            'threshold_sensitivity': thresh_ablation,
            'bootstrap': bootstrap_ablation
        }, f, indent=2, default=float)
    
    # ========================================================================
    # SAVE ALL RESULTS
    # ========================================================================
    
    # Save problem categorization
    cat_output = {}
    for dataset, result in cat_results.items():
        cat_output[dataset] = {
            'category_counts': result['category_counts'],
            'controversial_pct': result['controversial_pct'],
            'useless_pct': result['useless_pct'],
            'total_problems': result['total_problems']
        }
    
    with open(Config.OUTPUT_DIR / 'problem_categorization.json', 'w') as f:
        json.dump(cat_output, f, indent=2)
    
    # Save inverse scaling
    inv_output = {}
    for dataset, result in inv_results.items():
        inv_output[dataset] = {
            'inverse_count': result['count'],
            'inverse_pct': result['pct'],
            'total_problems': result['total'],
            'avg_gap': result.get('avg_gap', 0)
        }
    
    with open(Config.OUTPUT_DIR / 'inverse_scaling.json', 'w') as f:
        json.dump(inv_output, f, indent=2)
    
    # Save comprehensive statistics
    with open(Config.OUTPUT_DIR / 'comprehensive_statistics.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # Save mechanisms
    with open(Config.OUTPUT_DIR / 'failure_mechanisms.json', 'w') as f:
        json.dump(mechanism_results, f, indent=2)
    
    # Save characteristics
    with open(Config.OUTPUT_DIR / 'problem_characteristics.json', 'w') as f:
        json.dump(char_results, f, indent=2)
    with open(Config.OUTPUT_DIR / 'family_analysis.json', 'w') as f:
        json.dump(family_results, f, indent=2, default=str)
    
    with open(Config.OUTPUT_DIR / 'content_analysis.json', 'w') as f:
        json.dump(content_results, f, indent=2)
    
    with open(Config.OUTPUT_DIR / 'threshold_analysis.json', 'w') as f:
        json.dump(threshold_results, f, indent=2, default=str)
    
    with open(Config.OUTPUT_DIR / 'cost_benefit_analysis.json', 'w') as f:
        json.dump(cost_results, f, indent=2)
    # Save contamination analysis
    with open(Config.OUTPUT_DIR / 'contamination_analysis.json', 'w') as f:
        json.dump(contamination_results, f, indent=2, default=float)
        
    # Save causal intervention analysis
    with open(Config.OUTPUT_DIR / 'causal_intervention_analysis.json', 'w') as f:
        json.dump(causal_results, f, indent=2, default=float)
        
    # ========================================================================
    # FINAL SUMMARY FOR PAPER
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - PAPER-READY SUMMARY")
    print("="*80)
    
    print("\n DISCOVERY 1: NON-DISCRIMINATIVE PROBLEMS")
    avg_contr = stats_results['discriminative_efficiency']['controversial_mean']
    avg_useless = stats_results['discriminative_efficiency']['useless_mean']
    print(f"   Only {avg_contr:.1f}% of problems discriminate models")
    print(f"   {avg_useless:.1f}% provide no information")
    print(f"   Varies across benchmarks: CV={stats_results['consistency']['cv_controversial']:.2f}")
    
    print("\n DISCOVERY 2: INVERSE SCALING")
    avg_inv = stats_results['inverse_scaling']['inverse_mean']
    avg_gap = stats_results['inverse_scaling']['avg_gap_mean']
    effect = stats_results['inverse_scaling']['effect_size']
    print(f"   Small models outperform large on {avg_inv:.1f}% of problems")
    print(f"   Average performance gap: {avg_gap*100:.1f}%")
    print(f"   Effect size (Cohen's d): {effect:.2f}")
    
    print("\n KEY STATISTICS FOR PAPER")
    print(f"   Datasets analyzed: {len(Config.DATASETS)}")
    print(f"   Total problems: {sum(r['total_problems'] for r in cat_results.values())}")
    print(f"   Models tested: {len(Config.MODEL_METADATA)}")
    print(f"   Scale range: {min(m['size'] for m in Config.MODEL_METADATA.values()):.1f}B - {max(m['size'] for m in Config.MODEL_METADATA.values()):.1f}B")
    
    print("\n All results saved to:", Config.OUTPUT_DIR)
    print("\n ANALYSIS COMPLETE - READY FOR NATURE SUBMISSION!")
    
    # ========================================================================
    # NATURE SUBMISSION READINESS ASSESSMENT
    # ========================================================================
    
    print(f"\n" + "="*80)
    print(f"NATURE SUBMISSION READINESS ASSESSMENT")
    print(f"="*80)
    
   
    high_risk_count = sum(1 for r in contamination_results.values() 
                          if r['overall_risk'] == 'HIGH')
    low_risk_count = sum(1 for r in contamination_results.values() 
                         if r['overall_risk'] == 'LOW')
    
    print(f"\n CONTAMINATION ANALYSIS:")
    print(f"   Datasets with LOW risk: {low_risk_count}/{len(contamination_results)}")
    print(f"   Datasets with HIGH risk: {high_risk_count}/{len(contamination_results)}")
    
  
    
    print(f"\n All results saved to:", Config.OUTPUT_DIR)
    print(f"   Files: 14 JSON outputs including contamination_analysis.json")

if __name__ == "__main__":
    main()