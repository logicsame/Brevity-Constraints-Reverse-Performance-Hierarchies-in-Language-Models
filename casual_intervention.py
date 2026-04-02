
from cross_model_prober import CrossModelProber
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm
import pandas as pd

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
def main_causal_intervention():
    """
    CAUSAL INTERVENTION EXPERIMENT
    Re-evaluate inverse scaling problems under 3 conditions:
    1. Control: Original prompts (allow full reasoning)
    2. Brief: Force <50 word responses
    3. Direct: Force direct answers only
    """
    
    print(f"\n{'#'*80}")
    print(f"# CAUSAL INTERVENTION EXPERIMENT")
    print(f"# Testing overthinking hypothesis on inverse scaling problems")
    print(f"{'#'*80}\n")
    
    
    
    INVERSE_PROBLEMS = {
    "gsm8k": [40, 49, 61, 113, 121, 137, 211, 215, 222, 232, 235, 239, 248],
    "boolq": [2, 12, 21, 37, 42, 49, 66, 88, 102, 114, 119, 121, 122, 124, 136, 162, 174, 177, 182, 188, 200, 210, 224, 229, 241, 249, 254, 261, 262, 265, 278, 283, 292, 293],
    "arc-easy": [0, 9, 27, 38, 48, 54, 55, 63, 66, 71, 103, 104, 123, 127, 151, 173, 179, 184, 189, 210, 214, 215, 222, 232, 245, 253, 255, 280],
    "commonsenseqa": [10, 33, 57, 63, 75, 88, 91, 118, 135, 136, 140, 144, 172, 174, 185, 190, 199, 211, 216, 230, 231, 232, 241, 249, 265, 269, 270, 275, 299],
    "mmlu-stem": [11, 49, 61, 92, 102, 126, 155, 246, 250, 258, 282]
}
    
    print(f"✅ Loaded inverse scaling problem IDs:")
    total_problems = sum(len(ids) for ids in INVERSE_PROBLEMS.values())
    for dataset, ids in INVERSE_PROBLEMS.items():
        print(f"   {dataset:<20}: {len(ids)} problems")
    print(f"   TOTAL: {total_problems} inverse scaling problems\n")
    
    # ========== MODELS TO TEST ==========
    SMALL_MODELS = [
        # "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-3B-Instruct",
        # "databricks-qwen3-next-80b-a3b-instruct",
        # "databricks-meta-llama-3-1-405b-instruct"
    ]
  
    
    ALL_MODELS = SMALL_MODELS 
    
    CONDITIONS = ['control', 'brief', 'direct']
    # CONDITIONS = ['control', 'brief']
    
    DEBUG = False
    
    # ========== DATASET LOADERS ==========
    DATASET_LOADERS = {
        'gsm8k': (load_gsm8k, 'math'),
        'boolq': (load_boolq, 'reasoning'),
        'arc-easy': (load_arc_easy, 'commonsense'),
        'commonsenseqa': (load_commonsenseqa, 'commonsense'),
        'mmlu-stem': (load_mmlu, 'commonsense'),
    }
    
    # ========== RUN INTERVENTION FOR EACH DATASET ==========
    for dataset_name, (loader_func, task_type) in DATASET_LOADERS.items():
        
        if dataset_name not in INVERSE_PROBLEMS or len(INVERSE_PROBLEMS[dataset_name]) == 0:
            print(f"⚠️ Skipping {dataset_name}: No inverse scaling problems")
            continue
        
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Inverse problems: {len(INVERSE_PROBLEMS[dataset_name])}")
        print(f"Conditions: {CONDITIONS}")
        print(f"Models: {len(ALL_MODELS)}")
        
        # Load full dataset
        print(f"\n📚 Loading {dataset_name}...")
        all_samples, _ = loader_func(num_samples=10000)  # Load all
        
        # Filter to inverse scaling problems only
        inverse_sample_ids = set(INVERSE_PROBLEMS[dataset_name])
        inverse_samples = [s for s in all_samples if s['id'] in inverse_sample_ids]
        
        print(f"✅ Filtered to {len(inverse_samples)} inverse scaling problems")
        
        # Create output directory
        output_dir = Path(f"./causal_intervention_results/{dataset_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "raw_responses").mkdir(exist_ok=True)
        
        # ========== TEST EACH CONDITION ==========
        for condition in CONDITIONS:
            print(f"\n{'─'*60}")
            print(f"🧪 CONDITION: {condition.upper()}")
            print(f"{'─'*60}")
            
            # Create prober
            prober = CrossModelProber(
                model_names=ALL_MODELS,
                output_dir=str(output_dir),
                debug=DEBUG
            )
            
            # Probe all inverse problems with this condition
            all_responses = []
            
            for sample in tqdm(inverse_samples, desc=f"{dataset_name} - {condition}"):
                try:
                    # Probe with specific condition
                    sample_responses = prober.probe_sample(sample, task_type, condition)
                    all_responses.extend(sample_responses)
                except Exception as e:
                    print(f"❌ Error on sample {sample['id']}: {e}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame([asdict(r) for r in all_responses])
            
            # Add model size category
            df['model_size'] = df['model_name'].apply(
                lambda x: 'small' if x in SMALL_MODELS else 'large'
            )
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / "raw_responses" / f"{condition}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\n💾 Saved {len(df)} responses to: {output_file}")
            
            # Print summary
            print(f"\n📊 {condition.upper()} CONDITION RESULTS:")
            print(f"   Small models: {df[df['model_size']=='small']['is_correct'].mean():.1%}")
            print(f"   Large models: {df[df['model_size']=='large']['is_correct'].mean():.1%}")
            gap = (df[df['model_size']=='small']['is_correct'].mean() - 
                   df[df['model_size']=='large']['is_correct'].mean())
            print(f"   Gap: {gap:+.1%}")
            
            # Clean up models
            prober.cleanup_all_models()
    
    print(f"\n{'#'*80}")
    print(f"# CAUSAL INTERVENTION COMPLETE")
    print(f"{'#'*80}")
    print(f"\n✅ Results saved to: ./causal_intervention_results/")
    print(f"\n📊 Next step: Run analysis to compare conditions")

if __name__ == "__main__":
    # Login to Hugging Face (if needed)
    from huggingface_hub import login
    login(token="token_here")
    main_causal_intervention()
