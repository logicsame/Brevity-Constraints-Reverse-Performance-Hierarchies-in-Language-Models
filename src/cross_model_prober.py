from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from tqdm import tqdm
from model_manager import ModelManager
from reasoning_extractor import ReasoningTraceExtractor
from prompt_formater import PromptFormatter
from validator import MathValidator, ReasoningValidator, MultipleChoiceValidator
from validator import ModelResponse







# ============================================================================
# CROSS-MODEL PROBER
# ============================================================================

class CrossModelProber:
    """Probe multiple models with identical prompts"""

    def __init__(self, model_names: List[str], output_dir: str = "./cross_model_results",
                 debug: bool = False):
        self.model_names = model_names
        self.output_dir = Path(output_dir)
        self.debug = debug

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw_responses").mkdir(exist_ok=True)
        (self.output_dir / "divergence_analysis").mkdir(exist_ok=True)

        # Initialize validators
        self.math_validator = MathValidator(debug)
        self.reasoning_validator = ReasoningValidator()
        self.mc_validator = MultipleChoiceValidator()
        self.trace_extractor = ReasoningTraceExtractor(debug)

        # 🚀 CRITICAL FIX: Pre-load ALL models once
        self.model_managers = {}
        print("🚀 Pre-loading all models...")
        for model_name in self.model_names:
            try:
                self.model_managers[model_name] = ModelManager(model_name, debug=self.debug)
                print(f"✅ Pre-loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")

        print(f"📁 Output directory: {self.output_dir}")

    def probe_sample(self, sample: Dict, task_type: str) -> List[ModelResponse]:
        """
        Probe ALL models with the same sample - FIXED VERSION
        """
        question = sample['question']
        ground_truth = sample['ground_truth']
        sample_id = sample['id']
        dataset_name = sample['dataset']

        # Create identical prompt for all models
        prompt = PromptFormatter.format(question, task_type)

        if self.debug:
            print(f"\n{'='*80}")
            print(f"🔍 PROBING SAMPLE #{sample_id}")
            print(f"📝 QUESTION: {question}")
            print(f"✅ GROUND TRUTH: {ground_truth}")
            print(f"🎯 TASK TYPE: {task_type}")
            print(f"📋 PROMPT USED:\n{prompt}")
            print(f"{'='*80}\n")

        responses = []

        # 🚀 FIX: Use pre-loaded models instead of loading/unloading each time
        for model_name, manager in self.model_managers.items():
            try:
                # Generate response using pre-loaded model
                # Task-specific token limits
                TOKEN_LIMITS = {
                    'math': 400,
                    'reasoning': 200,  # BoolQ should be SHORT
                    'commonsense': 256,
                }
                max_tokens = TOKEN_LIMITS.get(task_type, 300)

                full_text, gen_time, in_tokens, out_tokens = manager.generate(
                    prompt,
                    max_new_tokens=max_tokens
                )

                # Extract just the generated part (remove prompt)
                generation = full_text[len(prompt):].strip()

                # Extract reasoning steps
                reasoning_steps = self.trace_extractor.extract_steps(generation, task_type)

                # Extract answer
                extracted_answer = self._extract_answer(
                    generation,
                    task_type,
                    ground_truth  # ← Add this parameter!
                )
                # Validate
                is_correct = self._validate_answer(extracted_answer, ground_truth, task_type)

                if self.debug:
                    print(f"🤖 MODEL: {model_name}")
                    print(f"📤 FULL GENERATION:\n{generation}")
                    print(f"🔍 EXTRACTED ANSWER: {extracted_answer}")
                    print(f"✅ CORRECT: {is_correct}")
                    print(f"🕒 GENERATION TIME: {gen_time:.2f}s")
                    print(f"📊 TOKENS: {in_tokens} in, {out_tokens} out")
                    print(f"🔢 REASONING STEPS ({len(reasoning_steps)} steps):")
                    for i, step in enumerate(reasoning_steps, 1):
                        print(f"   {i}. {step}")
                    print(f"{'-'*60}\n")

                # Create response object
                response = ModelResponse(
                    model_name=model_name,
                    sample_id=sample_id,
                    question=question,
                    ground_truth=ground_truth,
                    task_type=task_type,
                    dataset_name=dataset_name,
                    prompt_used=prompt,
                    full_generation=generation,
                    reasoning_trace=reasoning_steps,
                    extracted_answer=extracted_answer,
                    is_correct=is_correct,
                    generation_time=gen_time,
                    num_reasoning_steps=len(reasoning_steps),
                    timestamp=datetime.now().isoformat(),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens
                )

                responses.append(response)

            except Exception as e:
                print(f"❌ Error with {model_name} on sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return responses

    def _extract_answer(self, text: str, task_type: str, ground_truth: Any = None) -> Any:
        """Extract answer based on task type"""
        if task_type == "math":
            return self.math_validator.extract_answer(text, ground_truth)  # ← Pass ground_truth!
        elif task_type == "reasoning":
            return self.reasoning_validator.extract_answer(text)
        elif task_type == "commonsense":
            return self.mc_validator.extract_answer(text)
        else:
            return None

    def _validate_answer(self, predicted: Any, ground_truth: Any, task_type: str) -> bool:
        """Validate answer based on task type"""
        if task_type == "math":
            return self.math_validator.validate(predicted, ground_truth)
        elif task_type == "reasoning":
            return self.reasoning_validator.validate(predicted, ground_truth)
        elif task_type == "commonsense":
            return self.mc_validator.validate(predicted, ground_truth)
        else:
            return False

    def probe_dataset(self, samples: List[Dict], task_type: str) -> pd.DataFrame:
        """
        Probe all models on all samples.
        Returns DataFrame with all responses.
        """
        print(f"\n{'#'*80}")
        print(f"# CROSS-MODEL PROBING: {len(self.model_names)} models × {len(samples)} samples")
        print(f"{'#'*80}\n")

        all_responses = []

        for sample in tqdm(samples, desc="Probing samples"):
            sample_responses = self.probe_sample(sample, task_type)
            all_responses.extend(sample_responses)

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_responses])

        # Save raw responses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "raw_responses" / f"responses_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n💾 Responses saved to: {output_path}")

        # Print summary
        # Print summary
        self._print_summary(df)

        # Validate extraction quality
        self.validate_extraction_quality(df, "current_dataset")

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print(f"\n{'='*80}")
        print(f"📊 CROSS-MODEL SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total responses: {len(df)}")
        print(f"Models: {df['model_name'].nunique()}")
        print(f"Samples: {df['sample_id'].nunique()}")

        print(f"\n📈 Accuracy by model:")
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            accuracy = model_df['is_correct'].mean()
            avg_steps = model_df['num_reasoning_steps'].mean()
            avg_time = model_df['generation_time'].mean()

            print(f"  {model}:")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    Avg steps: {avg_steps:.1f}")
            print(f"    Avg time: {avg_time:.2f}s")


    def validate_extraction_quality(self, df: pd.DataFrame, dataset_name: str):
          """Check if extraction is working correctly"""

          print(f"\n{'='*80}")
          print(f"EXTRACTION QUALITY CHECK: {dataset_name}")
          print(f"{'='*80}\n")

          # Count failed extractions (None values)
          failed = df[df['extracted_answer'].isna()]

          print(f"Total responses: {len(df)}")
          print(f"Failed extractions: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")

          # Check for suspiciously low accuracy
          for model in df['model_name'].unique():
              model_data = df[df['model_name'] == model]
              acc = model_data['is_correct'].mean()

              if acc == 0.0:
                  print(f"\n⚠️ WARNING: {model.split('/')[-1]} has 0% accuracy on {dataset_name}!")
                  print(f"   This likely indicates extraction failure, not model failure")
                  print(f"   Sample outputs:")
                  for idx, row in model_data.head(2).iterrows():
                      print(f"\n   Sample {row['sample_id']}:")
                      print(f"   Output: {row['full_generation'][:150]}...")
                      print(f"   Extracted: {row['extracted_answer']}")
                      print(f"   Ground Truth: {row['ground_truth']}")

    def cleanup_all_models(self):
        """Clean up all loaded models at the end"""
        print("🧹 Cleaning up all models...")
        for model_name, manager in self.model_managers.items():
            try:
                manager.cleanup()
                print(f"✅ Cleaned up: {model_name}")
            except Exception as e:
                print(f"❌ Error cleaning up {model_name}: {e}")
        self.model_managers = {}