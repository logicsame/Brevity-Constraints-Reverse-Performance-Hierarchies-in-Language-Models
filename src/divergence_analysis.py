from typing import List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
from validator import DivergencePoint
import pandas as pd

# ============================================================================
# DIVERGENCE ANALYZER
# ============================================================================

class DivergenceAnalyzer:
    """Analyze where and how models' reasoning diverges"""

    def __init__(self, output_dir: Path, debug: bool = False):
        self.output_dir = output_dir
        self.debug = debug

    def analyze_divergence(self, df: pd.DataFrame) -> List[DivergencePoint]:
        """
        Find divergence points across all samples.
        """
        print(f"\n{'='*80}")
        print(f" DIVERGENCE ANALYSIS")
        print(f"{'='*80}\n")

        divergence_points = []

        # Group by sample
        for sample_id in df['sample_id'].unique():
            sample_df = df[df['sample_id'] == sample_id]

            if len(sample_df) < 2:
                continue  # Need at least 2 models

            # Get question
            question = sample_df.iloc[0]['question']

            # Compare reasoning traces
            sample_divergences = self._find_sample_divergences(sample_df, question)
            divergence_points.extend(sample_divergences)

        print(f"Found {len(divergence_points)} divergence points")

        # Save divergence analysis
        if divergence_points:
            divergence_df = pd.DataFrame([asdict(d) for d in divergence_points])
            output_path = self.output_dir / "divergence_analysis" / "divergence_points.csv"
            divergence_df.to_csv(output_path, index=False)
            print(f"💾 Divergence points saved to: {output_path}")

        return divergence_points

    def _find_sample_divergences(self, sample_df: pd.DataFrame,
                                 question: str) -> List[DivergencePoint]:
        """Find divergence points for a single sample across models"""
        sample_id = sample_df.iloc[0]['sample_id']
        divergences = []

        # Get all reasoning traces
        traces = {}
        for _, row in sample_df.iterrows():
            traces[row['model_name']] = row['reasoning_trace']

        # Find maximum number of steps
        max_steps = max(len(trace) for trace in traces.values())

        # Check each step for divergence
        for step_num in range(max_steps):
            step_texts = {}

            for model, trace in traces.items():
                if step_num < len(trace):
                    step_texts[model] = trace[step_num]

            if len(step_texts) < 2:
                continue

            # Check if steps are different
            unique_steps = list(set(step_texts.values()))

            if len(unique_steps) >= 2:
                # Found divergence!
                # Group models by their reasoning
                path_groups = {}
                for model, step_text in step_texts.items():
                    if step_text not in path_groups:
                        path_groups[step_text] = []
                    path_groups[step_text].append(model)

                # Take top 2 paths
                sorted_paths = sorted(path_groups.items(),
                                    key=lambda x: len(x[1]),
                                    reverse=True)

                if len(sorted_paths) >= 2:
                    path_a_text, path_a_models = sorted_paths[0]
                    path_b_text, path_b_models = sorted_paths[1]

                    # Count correct answers per path
                    path_a_correct = sum(
                        1 for m in path_a_models
                        if sample_df[sample_df['model_name'] == m]['is_correct'].iloc[0]
                    )
                    path_b_correct = sum(
                        1 for m in path_b_models
                        if sample_df[sample_df['model_name'] == m]['is_correct'].iloc[0]
                    )

                    # Calculate significance
                    significance = self._calculate_divergence_significance(
                        path_a_text, path_b_text
                    )

                    divergence = DivergencePoint(
                        sample_id=sample_id,
                        question=question,
                        step_number=step_num,
                        step_description=f"Step {step_num + 1}",
                        path_a_models=path_a_models,
                        path_b_models=path_b_models,
                        path_a_reasoning=path_a_text,
                        path_b_reasoning=path_b_text,
                        path_a_correct_count=path_a_correct,
                        path_b_correct_count=path_b_correct,
                        divergence_significance=significance
                    )

                    divergences.append(divergence)

        return divergences

    def _calculate_divergence_significance(self, text_a: str, text_b: str) -> float:
        """
        Calculate how different two reasoning steps are.
        Returns value between 0 (identical) and 1 (completely different).
        """
        # Simple Jaccard similarity on words
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 1.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        similarity = intersection / union if union > 0 else 0
        divergence = 1.0 - similarity

        return divergence