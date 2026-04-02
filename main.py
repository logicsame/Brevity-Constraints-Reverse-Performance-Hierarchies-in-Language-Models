from pathlib import Path
from typing import List, Dict, Tuple
from cross_model_prober import CrossModelProber
from divergence_analysis import DivergenceAnalyzer
from utils.load_dataset import (
    load_gsm8k, 
    load_commonsenseqa,
    load_boolq,
    load_arc_easy,
    load_mmlu
)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main cross-model cognitive archaeology pipeline - MULTI-DATASET VERSION"""

    # ========== CONFIGURATION ==========
    MODELS_TO_TEST = [
        # "01-ai/Yi-1.5-6B-Chat",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "google/gemma-2-9b-it",
        # "meta-llama/Llama-2-13b-hf"
        # "stabilityai/stablelm-zephyr-3b",
        # "mistralai/Mistral-Small-24B-Instruct-2501",
        # "google/gemma-3-27b-it"
        # "deepseek-ai/deepseek-llm-67b-base"
        # "meta-llama/Meta-Llama-3-70B"
        # "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # "mistralai/Mixtral-8x22B-v0.1"
        # "mistralai/Mistral-Nemo-Instruct-2407"
        # "allenai/longformer-base-4096"
        "tiiuae/falcon-7b"
    ]

    NUM_SAMPLES_PER_DATASET = 300 # ← Adjust this
    DEBUG = False

    # ========== DATASETS TO TEST ==========
    DATASETS = [
        ("GSM8K", load_gsm8k),
        ("CommonsenseQA", load_commonsenseqa),
        ("BoolQ", load_boolq),
        ("ARC-Easy", load_arc_easy),
        ("MMLU-STEM", load_mmlu),
    ]

    # ========== RUN ON ALL DATASETS ==========
    all_results = {}

    for dataset_name, loader_func in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}\n")

        # Create dataset-specific output directory
        output_dir = f"./cross_model_archaeology/{dataset_name.lower()}"

        prober = CrossModelProber(
            model_names=MODELS_TO_TEST,
            output_dir=output_dir,
            debug=DEBUG
        )

        try:
            # ========== LOAD DATASET ==========
            samples, task_type = loader_func(num_samples=NUM_SAMPLES_PER_DATASET)

            # ========== PROBE ALL MODELS ==========
            # ========== PROBE ALL MODELS ==========
            responses_df = prober.probe_dataset(samples, task_type)

            # Validate extraction quality
            prober.validate_extraction_quality(responses_df, dataset_name)

            # ========== ANALYZE DIVERGENCE ==========
            analyzer = DivergenceAnalyzer(output_dir=Path(output_dir), debug=DEBUG)
            divergence_points = analyzer.analyze_divergence(responses_df)

            # Store results
            all_results[dataset_name] = {
                'responses': responses_df,
                'divergences': divergence_points,
                'task_type': task_type
            }

            print(f"\n Completed {dataset_name}")
            print(f"   Responses: {len(responses_df)}")
            print(f"   Divergences: {len(divergence_points)}")

        except Exception as e:

            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up models after each dataset
            prober.cleanup_all_models()

    # ========== CROSS-DATASET COMPARISON ==========
    print(f"\n{'#'*80}")
    print(f"# CROSS-DATASET SUMMARY")
    print(f"{'#'*80}\n")

    for dataset_name, results in all_results.items():
        df = results['responses']
        print(f"\n{dataset_name}:")
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            accuracy = model_df['is_correct'].mean()
            print(f"  {model}: {accuracy:.1%}")

    print(f"\n{'#'*80}")
    print(f"# ALL DATASETS COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    from huggingface_hub import login
    login(token="token_here")

    main() 