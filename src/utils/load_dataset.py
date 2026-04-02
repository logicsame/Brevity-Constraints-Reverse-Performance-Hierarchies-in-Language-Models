from typing import List, Dict, Tuple
import re
from datasets import load_dataset




# ============================================================================
# DATASET LOADER
# ============================================================================

def load_gsm8k(num_samples: int = 50, split: str = "test") -> Tuple[List[Dict], str]:
    """Load GSM8K dataset"""
    print(f"📚 Loading GSM8K ({split} split, {num_samples} samples)...")

    dataset = load_dataset("gsm8k", "main", split=split)
    samples = []

    for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        question = item['question']
        answer_text = item['answer']

        # Extract numerical answer
        match = re.search(r'####\s*(\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        if match:
            ground_truth = float(match.group(1).replace(',', ''))
            samples.append({
                'id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'dataset': 'gsm8k'
            })

    print(f"✅ Loaded {len(samples)} samples from GSM8K")
    return samples, "math"

# ============================================================================
# DATASET LOADERS - ADD THESE
# ============================================================================

def load_strategyqa(num_samples: int = 50, split: str = "train") -> Tuple[List[Dict], str]:
    """Load StrategyQA dataset"""
    print(f"📚 Loading StrategyQA ({split} split, {num_samples} samples)...")

    dataset = load_dataset("wics/strategy-qa", split=split)
    samples = []

    for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        question = item['question']
        # StrategyQA has boolean answers
        ground_truth = item['answer']  # True/False

        samples.append({
            'id': idx,
            'question': question,
            'ground_truth': ground_truth,
            'dataset': 'strategyqa'
        })

    print(f"✅ Loaded {len(samples)} samples from StrategyQA")
    return samples, "reasoning"  # ← Task type is "reasoning" (yes/no)

def load_mmlu(num_samples: int = 50, split: str = "test", subject: str = "stem") -> Tuple[List[Dict], str]:
    """Load MMLU dataset - STEM subjects only"""
    print(f"📚 Loading MMLU-STEM ({split} split, {num_samples} samples)...")

    # MMLU STEM subjects
    STEM_SUBJECTS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'electrical_engineering',
        'elementary_mathematics',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_mathematics',
        'high_school_physics',
        'high_school_statistics',
        'machine_learning',
    ]

    samples = []
    samples_per_subject = max(1, num_samples // len(STEM_SUBJECTS))

    for subject in STEM_SUBJECTS[:num_samples // samples_per_subject + 1]:
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)

            for idx, item in enumerate(dataset.select(range(min(samples_per_subject, len(dataset))))):
                question = item['question']
                choices = item['choices']  # List of 4 choices
                answer_idx = item['answer']  # 0-3

                # Format question with choices
                formatted_question = f"{question}\n"
                for i, choice_text in enumerate(choices):
                    label = chr(65 + i)  # A, B, C, D
                    formatted_question += f"{label}) {choice_text}\n"

                # Convert answer index to letter
                answer_letter = chr(65 + answer_idx).lower()  # 'a', 'b', 'c', 'd'

                samples.append({
                    'id': len(samples),
                    'question': formatted_question,
                    'ground_truth': answer_letter,
                    'dataset': f'mmlu_{subject}'
                })

                if len(samples) >= num_samples:
                    break
        except Exception as e:
            print(f"⚠️ Failed to load {subject}: {e}")
            continue

        if len(samples) >= num_samples:
            break

    print(f"✅ Loaded {len(samples)} samples from MMLU-STEM")
    return samples, "commonsense"  # ← Multiple choice task type

def load_commonsenseqa(num_samples: int = 50, split: str = "validation") -> Tuple[List[Dict], str]:
    """Load CommonsenseQA dataset"""
    print(f"📚 Loading CommonsenseQA ({split} split, {num_samples} samples)...")

    dataset = load_dataset("tau/commonsense_qa", split=split)
    samples = []

    for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']  # ['A', 'B', 'C', 'D', 'E']
        answer_key = item['answerKey']

        # Format question with choices
        formatted_question = f"{question}\n"
        for label, choice_text in zip(labels, choices):
            formatted_question += f"{label}) {choice_text}\n"

        samples.append({
            'id': idx,
            'question': formatted_question,
            'ground_truth': answer_key.lower(),  # 'a', 'b', 'c', etc.
            'dataset': 'commonsense_qa'
        })

    print(f"✅ Loaded {len(samples)} samples from CommonsenseQA")
    return samples, "commonsense"  # ← Task type is "commonsense" (multiple choice)


def load_boolq(num_samples: int = 50, split: str = "validation") -> Tuple[List[Dict], str]:
    """Load BoolQ dataset"""
    print(f"📚 Loading BoolQ ({split} split, {num_samples} samples)...")

    dataset = load_dataset("google/boolq", split=split)
    samples = []

    for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        passage = item['passage']
        question = item['question']
        answer = item['answer']  # True/False

        # Combine passage and question
        full_question = f"Passage: {passage}\n\nQuestion: {question}"

        samples.append({
            'id': idx,
            'question': full_question,
            'ground_truth': answer,
            'dataset': 'boolq'
        })

    print(f"✅ Loaded {len(samples)} samples from BoolQ")
    return samples, "reasoning"  # ← Same as StrategyQA


def load_arc_easy(num_samples: int = 50, split: str = "test") -> Tuple[List[Dict], str]:
    """Load ARC-Easy dataset"""
    print(f"📚 Loading ARC-Easy ({split} split, {num_samples} samples)...")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    samples = []

    for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']  # ['A', 'B', 'C', 'D']
        answer_key = item['answerKey']

        # Format question with choices
        formatted_question = f"{question}\n"
        for label, choice_text in zip(labels, choices):
            formatted_question += f"{label}) {choice_text}\n"

        samples.append({
            'id': idx,
            'question': formatted_question,
            'ground_truth': answer_key.lower(),  # Convert to lowercase
            'dataset': 'arc_easy'
        })

    print(f"✅ Loaded {len(samples)} samples from ARC-Easy")
    return samples, "commonsense"  # ← Multiple choice
