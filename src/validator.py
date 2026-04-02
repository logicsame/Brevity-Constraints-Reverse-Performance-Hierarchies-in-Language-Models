import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile

import torch
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.disable()
# torch._dynamo.config.ignore_logger_methods = ["warning_once", "warning", "info", "debug"]
import transformers
transformers.logging.set_verbosity_error()

import re
from typing import Any, Tuple


import gzip
import bz2
import zlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats

from dataclasses import dataclass
import re
import ast
from collections import defaultdict
from typing import Any
import re
from typing import Any, Tuple, Optional


class Validator:
    """
Research-grade answer extraction and validation system with multiple fallback strategies.

This class provides robust extraction of answers from verbose model outputs across
different task types (numeric, multiple choice, boolean, code). It handles the
challenge of extracting the actual answer when models produce lengthy reasoning
chains, explanations, or ambiguous outputs.

Core Philosophy:
    Models often embed answers within extensive reasoning chains. This validator
    uses a hierarchical strategy system to find and extract the intended answer,
    prioritizing explicit answer markers over implicit patterns.

Key Improvements over Basic Validation:
    1. Task-specific extraction strategies
    2. Comprehensive debugging and statistics tracking
    3. Handles verbose model outputs with reasoning chains
    4. Extracts final answers from multi-step reasoning
    5. Convention-aware multiple choice handling (0-based, 1-based, letter-based)

Supported Task Types:
    - exact_match: Direct string comparison
    - numeric: Numeric answer extraction with tolerance
    - contains: Substring matching (primarily for BoolQ yes/no)
    - multiple_choice: Letter/number choice extraction with normalization
    - code: Code block extraction with syntax validation

Numeric Extraction Strategies (Priority Order):
    1. "Final Answer: X" patterns (highest priority)
    2. LaTeX \\boxed{X} format
    3. Explicit markers ("answer is", "therefore")
    4. Last number in final sentences
    5. Last number in output (fallback, with position filtering)

Multiple Choice Extraction Strategies:
    1. Very short outputs (single character)
    2. Explicit answer patterns ("answer: A", "choice is B")
    3. Negation detection ("not A", "cannot be B")
    4. First sentence with context filtering (avoids false positives)
    5. Last occurrence in final 30% of output
    6. First occurrence (fallback)

Multiple Choice Conventions:
    - HellaSwag: 0-based indexing (0,1,2,3)
    - WinoGrande: 1-based indexing (1,2)
    - ARC-Easy, CommonsenseQA, OpenBookQA: Letter-based (a,b,c,d,e)
    - Auto-detection when task type unknown

Boolean (Yes/No) Strategies:
    1. Explicit patterns ("answer is yes/no")
    2. First occurrence (when both yes and no appear, takes first)
    3. Only yes or only no detection

Attributes:
    use_llm_fallback (bool): Whether to use LLM for extraction fallback (not implemented)
    debug (bool): Enable detailed debugging output for extraction process
    validators (Dict): Mapping of task types to validation functions
    extraction_stats (Dict): Statistics tracking successful/failed extractions by strategy

Methods:
    validate(output, ground_truth, task_type, tolerance, task_name):
        Main validation entry point

    _exact_match(output, ground_truth, tolerance):
        Direct string comparison after normalization

    _numeric_match(output, ground_truth, tolerance):
        Extract and validate numeric answers with multiple strategies

    _contains_match(output, ground_truth, tolerance):
        Substring matching with special yes/no handling

    _multiple_choice_match(output, ground_truth, tolerance, task_name):
        Extract multiple choice answers with convention-aware normalization

    _code_match(output, ground_truth, tolerance):
        Extract code blocks and validate syntax

    _track_strategy(strategy):
        Track which extraction strategy was successfully used

Return Format:
    All validation methods return: Tuple[bool, Optional[Any]]
    - bool: Whether the extracted answer matches ground truth
    - Optional[Any]: The extracted answer (None if extraction failed)

Debug Mode:
    When debug=True, prints detailed information:
    - Ground truth and output preview
    - Each strategy attempted
    - Matches found at each step
    - Selected answer and reasoning
    - Success/failure status with difference metrics

Statistics Tracking:
    extraction_stats tracks:
    - total: Total validation attempts
    - successful: Successful extractions
    - failed: Failed extractions
    - by_strategy: Count of successful uses per strategy

Example:
    >>> validator = RobustValidator(debug=True)
    >>>
    >>> # Numeric validation
    >>> is_correct, extracted = validator.validate(
    ...     output="Let me calculate: 5 + 3 = 8. Final Answer: 8",
    ...     ground_truth=8,
    ...     task_type='numeric',
    ...     tolerance=0.01
    ... )
    >>> print(f"Correct: {is_correct}, Extracted: {extracted}")

    >>> # Multiple choice validation
    >>> is_correct, extracted = validator.validate(
    ...     output="After analyzing all options, the answer is B.",
    ...     ground_truth='b',
    ...     task_type='multiple_choice',
    ...     task_name='commonsense_qa'
    ... )

    >>> # Check statistics
    >>> print(validator.extraction_stats)

Use Cases:
    - Evaluating LLM outputs on benchmark tasks
    - Extracting answers from chain-of-thought reasoning
    - Handling diverse output formats from different models
    - Research on answer extraction reliability
    - Building robust evaluation pipelines

Robustness Features:
    - Handles comma-separated numbers (e.g., "1,000")
    - Position-based filtering to avoid extraction from problem statement
    - Context-aware filtering to avoid false positives (e.g., "a process")
    - Negation detection to avoid extracting eliminated options
    - Fallback strategies when explicit markers are missing
    - Empty/short output handling
"""

    def __init__(self, use_llm_fallback: bool = False, debug: bool = False):  # ← FIXED: Added debug
        self.use_llm_fallback = use_llm_fallback
        self.debug = debug  # ← ADDED

        self.validators = {
            'exact_match': self._exact_match,
            'numeric': self._numeric_match,
            'contains': self._contains_match,
            'multiple_choice': self._multiple_choice_match,
            'code': self._code_match,
        }

        # ← ADDED: Track extraction statistics
        self.extraction_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'by_strategy': {}
        }

    def _track_strategy(self, strategy: str):  # ← ADDED: Track which strategy was used
        """Track which extraction strategy was used"""
        if strategy not in self.extraction_stats['by_strategy']:
            self.extraction_stats['by_strategy'][strategy] = 0
        self.extraction_stats['by_strategy'][strategy] += 1

    def validate(self, output: str, ground_truth: Any, task_type: str = 'numeric',
                 tolerance: float = 0.01, task_name: str = None) -> Tuple[bool, Optional[str]]:
        """Validate output against ground truth"""
        if task_type not in self.validators:
            raise ValueError(f"Unknown validation type: {task_type}")

        # FIXED: Return the result directly (removed duplicate call)
        if task_type == 'multiple_choice':
            return self.validators[task_type](output, ground_truth, tolerance, task_name)
        else:
            return self.validators[task_type](output, ground_truth, tolerance)

    def _exact_match(self, output: str, ground_truth: Any, tolerance: float) -> Tuple[bool, str]:
        output_clean = output.strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        return (output_clean == gt_clean, output_clean)

    def _numeric_match(self, output: str, ground_truth: Any, tolerance: float) -> Tuple[bool, Optional[float]]:
        """Extract and validate numeric answers"""

        # ADD THIS BLOCK FOR DETAILED DEBUG
        if self.debug:
            print(f"\n{'='*80}")
            print(f"[NUMERIC VALIDATOR DEBUG - DETAILED]")
            print(f"{'='*80}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Output: '{output}'")
            print(f"{'='*80}\n")

        # Handle empty outputs
        if not output or len(output.strip()) < 5:
            if self.debug:
                print(f"⚠️ Output too short or empty: '{output}'")
            return (False, None)

        # Convert ground truth
        try:
            target = float(ground_truth)
            if self.debug:
                print(f"✓ Target value: {target}")
        except:
            if self.debug:
                print(f"⚠️ Ground truth '{ground_truth}' is not numeric")
            return (False, None)

        output_lower = output.lower()

        # ================================================================
        # STRATEGY 1: "Final Answer: X" format (HIGHEST PRIORITY)
        # ================================================================
        if self.debug:
            print(f"\n🔍 Trying Strategy 1: 'Final Answer:' pattern...")

        final_answer_patterns = [
            r'final\s+answer\s*[:\-]\s*(?:.*?)(\d+(?:,\d{3})*(?:\.\d+)?)(?:\D*)$',
        ]

        for pattern in final_answer_patterns:
            matches = re.findall(pattern, output_lower)
            if matches:
                strategy = "final_answer_explicit"
                try:
                    number_str = matches[-1].replace(',', '')
                    predicted = float(number_str)
                    is_correct = abs(predicted - target) <= tolerance
                    self._track_strategy(strategy)

                    if self.debug:
                        print(f"✓✓✓ Strategy: {strategy}")
                        print(f"    All matches found: {matches}")
                        print(f"    Selected (last): '{matches[-1]}'")
                        print(f"    Extracted number: {predicted}")
                        print(f"    Target: {target}")
                        print(f"    Match: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                        print(f"    Difference: {abs(predicted - target)}")

                    return (is_correct, predicted)
                except Exception as e:
                    if self.debug:
                        print(f"✗ Failed to parse: {e}")
                    continue

        # ================================================================
        # STRATEGY 1b: Alternative "Final Answer" pattern
        # ================================================================
        if self.debug:
            print(f"\n🔍 Trying Strategy 1b: 'Final Answer' full context...")

        final_answer_full_patterns = [
            r'final\s+answer\s*[:\-]\s*(.*?)(?:\n|$)',
        ]

        for pattern in final_answer_full_patterns:
            matches = re.findall(pattern, output_lower)
            if matches:
                strategy = "final_answer_full_context"
                try:
                    answer_text = matches[-1].strip()
                    numbers_in_context = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)

                    if self.debug:
                        print(f"✓ Found 'Final Answer' context")
                        print(f"    Context: '{answer_text}'")
                        print(f"    Numbers found: {numbers_in_context}")

                    if numbers_in_context:
                        number_str = numbers_in_context[-1].replace(',', '')
                        predicted = float(number_str)
                        is_correct = abs(predicted - target) <= tolerance
                        self._track_strategy(strategy)

                        if self.debug:
                            print(f"✓✓✓ Strategy: {strategy}")
                            print(f"    Selected (last): '{numbers_in_context[-1]}'")
                            print(f"    Extracted number: {predicted}")
                            print(f"    Target: {target}")
                            print(f"    Match: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

                        return (is_correct, predicted)
                except Exception as e:
                    if self.debug:
                        print(f"✗ Failed to parse: {e}")
                    continue

        # ================================================================
        # STRATEGY 2: LaTeX boxed format
        # ================================================================
        if self.debug:
            print(f"\n🔍 Trying Strategy 2: LaTeX \\boxed{{}} format...")

        boxed_patterns = [
            r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}',
        ]

        for pattern in boxed_patterns:
            matches = re.findall(pattern, output)
            if matches:
                if self.debug:
                    print(f"✓ Found \\boxed{{}} matches: {matches}")

                strategy = "latex_boxed"
                try:
                    number_str = matches[-1].replace(',', '')
                    predicted = float(number_str)
                    is_correct = abs(predicted - target) <= tolerance
                    self._track_strategy(strategy)

                    if self.debug:
                        print(f"✓✓✓ Strategy: {strategy} - Found: {predicted}")

                    return (is_correct, predicted)
                except:
                    continue

        # ================================================================
        # STRATEGY 3: Other explicit markers
        # ================================================================
        if self.debug:
            print(f"\n🔍 Trying Strategy 3: Other explicit markers...")

        other_explicit_patterns = [
            r'(?:the\s+)?answer\s+is\s*[:\-]?\s*.*?(\d+(?:,\d{3})*(?:\.\d+)?)(?:\D|$)',
            r'therefore[,\s]+(?:the\s+)?answer\s+is\s*[:\-]?\s*.*?(\d+(?:,\d{3})*(?:\.\d+)?)(?:\D|$)',
            r'so[,\s]+(?:the\s+)?answer\s+is\s*[:\-]?\s*.*?(\d+(?:,\d{3})*(?:\.\d+)?)(?:\D|$)',
            r'####\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
        ]

        for pattern in other_explicit_patterns:
            matches = re.findall(pattern, output_lower)
            if matches:
                if self.debug:
                    print(f"✓ Found matches with pattern '{pattern[:30]}...': {matches}")

                strategy = "explicit_marker"
                try:
                    number_str = matches[-1].replace(',', '')
                    predicted = float(number_str)
                    is_correct = abs(predicted - target) <= tolerance
                    self._track_strategy(strategy)

                    if self.debug:
                        print(f"✓✓✓ Strategy: {strategy} - Found: {predicted}")

                    return (is_correct, predicted)
                except:
                    continue

        # Strategy 4: Last number in final sentences
        sentences = [s.strip() for s in output.split('.') if s.strip()]
        if sentences:
            final_text = '. '.join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]

            # NEW: If text is repetitive (same pattern >3 times), use FIRST sentence only
            if final_text.count('=') > 5:  # Likely repetitive
                final_text = sentences[0] if sentences else final_text

            number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
            matches = re.findall(number_pattern, final_text)

            if self.debug:
                print(f"    Final text: '{final_text[:100]}...'")
                print(f"    Numbers found: {matches}")

            if matches:
                strategy = "final_sentence_number"
                try:
                    number_str = matches[-1].replace(',', '')
                    predicted = float(number_str)

                    if 0 <= predicted <= 1000000:
                        is_correct = abs(predicted - target) <= tolerance
                        self._track_strategy(strategy)

                        if self.debug:
                            print(f"✓✓✓ Strategy: {strategy} - Found: {predicted}")

                        return (is_correct, predicted)
                except:
                    pass

        # ================================================================
        # STRATEGY 5: Last number (FALLBACK)
        # ================================================================
        if self.debug:
            print(f"\n🔍 Trying Strategy 5: Last number fallback...")

        number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
        all_matches = re.findall(number_pattern, output)

        if self.debug:
            print(f"    All numbers in output: {all_matches}")

        if all_matches:
            strategy = "last_number_fallback"
            try:
                number_str = all_matches[-1].replace(',', '')
                predicted = float(number_str)

                last_match_pos = output.rfind(all_matches[-1])
                position_ratio = last_match_pos / len(output) if len(output) > 0 else 0

                if self.debug:
                    print(f"    Last number: '{all_matches[-1]}' at position {last_match_pos}")
                    print(f"    Position ratio: {position_ratio:.2%} (need >60%)")

                if position_ratio >= 0.6:
                    is_correct = abs(predicted - target) <= tolerance
                    self._track_strategy(strategy)

                    if self.debug:
                        print(f"⚠️ Strategy: {strategy} (fallback) - Found: {predicted}")

                    return (is_correct, predicted)
            except:
                pass

        # Failed to extract
        if self.debug:
            print(f"\n✗✗✗ ALL STRATEGIES FAILED")
            print(f"    No suitable numeric answer found in output")

        return (False, None)

    def _contains_match(self, output: str, ground_truth: Any, tolerance: float) -> Tuple[bool, str]:
      """ENHANCED BoolQ handling"""

      if not output or len(output.strip()) < 2:
          if self.debug:
              print(f"⚠️ Output too short or empty: '{output}'")
          return (False, None)

      # Strip markdown formatting before processing
      output_clean = output.strip().lower()
      output_clean = re.sub(r'\*\*', '', output_clean)  # Remove bold markdown **text**
      output_clean = re.sub(r'__', '', output_clean)    # Remove bold __text__
      output_clean = re.sub(r'\*', '', output_clean)    # Remove italic *text*

      gt_str = str(ground_truth).lower().strip()

      # ADD THIS BLOCK FOR BETTER DEBUG
      if self.debug:
          print(f"\n{'='*80}")
          print(f"[BOOLQ VALIDATOR DEBUG]")
          print(f"{'='*80}")
          print(f"Ground Truth: '{gt_str}'")
          print(f"Output (full): '{output_clean}'")  # ← Changed to show full output
          print(f"{'='*80}\n")

      # Special handling for yes/no (BoolQ)
      if gt_str in ['yes', 'no']:
          # ============================================================
          # CRITICAL FIX: Search in ENTIRE output, prioritizing the END
          # ============================================================

          # STRATEGY 1: Look for explicit answer markers (HIGHEST PRIORITY)
          explicit_patterns = [
              r'final\s+answer\s*[:\-]\s*(yes|no)',
              r'answer\s*[:\-]\s*(yes|no)',
              r'the\s+answer\s+(?:is|to\s+the\s+question\s+is)\s+(yes|no)',
              r'therefore[,\s]+(?:the\s+)?answer\s+is\s+(yes|no)',
          ]

          for pattern in explicit_patterns:
              match = re.search(pattern, output_clean)
              if match:
                  extracted = match.group(1)
                  if self.debug:
                      print(f"✓ Strategy: explicit_pattern - Found '{extracted}'")
                  return (extracted == gt_str, extracted)


          # STRATEGY 1.5: Check FIRST sentence (for direct answers like "No" or "Yes")
          sentences = output_clean.split('.')
          sentences = [s.strip() for s in sentences if s.strip()]

          if sentences:
              first_sentence = sentences[0].strip()

              # Check if first sentence has yes XOR no (not both)
              has_yes = re.search(r'\byes\b', first_sentence) is not None
              has_no = re.search(r'\bno\b', first_sentence) is not None

              if has_yes and not has_no:
                  if self.debug:
                      print(f"✓ Strategy: first_sentence_yes")
                  return ('yes' == gt_str, 'yes')

              if has_no and not has_yes:
                  if self.debug:
                      print(f"✓ Strategy: first_sentence_no")
                  return ('no' == gt_str, 'no')

          # STRATEGY 2: Look in LAST sentence only (avoid false positives)
          sentences = output_clean.split('.')
          sentences = [s.strip() for s in sentences if s.strip()]

          if sentences:
              last_sentence = sentences[-1].strip()

              # Check if last sentence has yes XOR no (not both)
              has_yes = re.search(r'\byes\b', last_sentence) is not None
              has_no = re.search(r'\bno\b', last_sentence) is not None

              if has_yes and not has_no:
                  if self.debug:
                      print(f"✓ Strategy: last_sentence_yes")
                  return ('yes' == gt_str, 'yes')

              if has_no and not has_yes:
                  if self.debug:
                      print(f"✓ Strategy: last_sentence_no")
                  return ('no' == gt_str, 'no')

          # STRATEGY 3: Count in FINAL 20% only
          split_point = int(len(output_clean) * 0.8)
          final_portion = output_clean[split_point:]

          yes_count = len(re.findall(r'\byes\b', final_portion))
          no_count = len(re.findall(r'\bno\b', final_portion))

          if yes_count > no_count:
              if self.debug:
                  print(f"✓ Strategy: final_portion_count - yes={yes_count}, no={no_count}")
              return ('yes' == gt_str, 'yes')
          elif no_count > yes_count:
              if self.debug:
                  print(f"✓ Strategy: final_portion_count - yes={yes_count}, no={no_count}")
              return ('no' == gt_str, 'no')

          # Failed to extract
          if self.debug:
              print(f"✗ FAILED - No yes/no found in output")

          return (False, None)

      # For other contains tasks, use simple substring match
      is_correct = gt_str in output_clean
      return (is_correct, gt_str if is_correct else None)

    def _multiple_choice_match(self, output: str, ground_truth: Any, tolerance: float, task_name: str = None) -> Tuple[bool, str]:
        """
        FIXED v6: Task-aware multiple choice validation with correct normalization

        Handles different indexing conventions:
        - HellaSwag: 0-based (0,1,2,3)
        - WinoGrande: 1-based (1,2)
        - ARC-Easy, CommonsenseQA, OpenBookQA: letters (a,b,c,d,e)
        """

        if not output:
            if self.debug:
                print(f"⚠️ Output empty")
            return (False, None)

        output_clean = output.strip().lower()
        gt_str = str(ground_truth).strip().lower()

        if self.debug:
            print(f"🔍 Multiple Choice Validation - Ground Truth: '{gt_str}', Task: '{task_name}'")
            print(f"🔍 Output: '{output_clean[:100]}...'")

        # ================================================================
        # STEP 1: Determine task-specific indexing convention
        # ================================================================
        TASK_CONVENTIONS = {
            'hellaswag': '0-based',      # 0,1,2,3
            'winogrande': '1-based',     # 1,2
            'arc_easy': 'letter',        # a,b,c,d
            'arc-easy': 'letter',        # Handle both naming styles
            'commonsense_qa': 'letter',  # a,b,c,d,e
            'commonsenseqa': 'letter',   # Handle both naming styles
            'openbookqa': 'letter',      # a,b,c,d
        }

        convention = TASK_CONVENTIONS.get(task_name, 'auto-detect')

        # Auto-detect convention from ground truth if task unknown
        if convention == 'auto-detect':
            if gt_str in ['0', '1', '2', '3', '4']:
                convention = '0-based'
            elif gt_str in ['a', 'b', 'c', 'd', 'e']:
                convention = 'letter'
            else:
                # Fallback: assume 1-based for numeric GT
                convention = '1-based' if gt_str.isdigit() else 'letter'

        if self.debug:
            print(f"🔍 Convention detected: {convention}")

        # ================================================================
        # STEP 2: FIXED - Define normalization based on convention AND ground truth
        # ================================================================
        def normalize_to_gt_format(extracted: str) -> str:
            """Convert extracted answer to ground truth format"""
            extracted = extracted.strip().lower()

            if convention == '0-based':
                # Ground truth is 0-based number (0,1,2,3,4)
                # Convert letters to 0-based numbers
                letter_to_0based = {'a': '0', 'b': '1', 'c': '2', 'd': '3', 'e': '4'}
                if extracted in letter_to_0based:
                    return letter_to_0based[extracted]
                return extracted

            elif convention == '1-based':
                # Ground truth is 1-based number (1,2,3,4,5)
                # Convert letters to 1-based numbers
                letter_to_1based = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5'}
                if extracted in letter_to_1based:
                    return letter_to_1based[extracted]
                return extracted

            else:  # letter convention
                # Ground truth is a letter (a,b,c,d,e)
                # If extracted is a letter, return as-is
                if extracted in ['a', 'b', 'c', 'd', 'e']:
                    return extracted

                # If extracted is a number, convert to letter
                # CRITICAL FIX: For letter convention, assume 0-based number input
                # (i.e., "0"→"a", "1"→"b", "2"→"c", etc.)
                if extracted in ['0', '1', '2', '3', '4', '5']:
                    number_to_letter = {
                        '0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e', '5': 'f'
                    }
                    return number_to_letter.get(extracted, extracted)

                return extracted

        # ================================================================
        # STRATEGY 1: Very short outputs (single character)
        # ================================================================
        if len(output_clean) <= 3 and output_clean in ['a', 'b', 'c', 'd', 'e', '0', '1', '2', '3', '4', '5']:
            strategy = "very_short_output"
            self._track_strategy(strategy)

            extracted = output_clean
            normalized = normalize_to_gt_format(extracted)

            if self.debug:
                print(f"✓ Strategy: {strategy} - Found: '{extracted}' → Normalized: '{normalized}'")

            return (normalized == gt_str, extracted)

        # ================================================================
        # STRATEGY 2: Explicit answer patterns
        # ================================================================
        explicit_patterns = [
            r'answer\s*[:\-]\s*([abcde012345])',
            r'answer\s+is\s+([abcde012345])',
            r'correct\s+answer\s*[:\-]?\s*([abcde012345])',
            r'choice\s*[:\-]\s*([abcde012345])',
            r'option\s*[:\-]\s*([abcde012345])',
            r'select\s+([abcde012345])',
            r'pick\s+([abcde012345])',
            r'choose\s+([abcde012345])',
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, output_clean)
            if match:
                extracted = match.group(1)
                normalized = normalize_to_gt_format(extracted)
                strategy = "explicit_answer"
                self._track_strategy(strategy)

                if self.debug:
                    print(f"✓ Strategy: {strategy} - Found: '{extracted}' → Normalized: '{normalized}'")

                return (normalized == gt_str, extracted)

        # ================================================================
        # STRATEGY 3: Check for negation patterns (IMPORTANT!)
        # ================================================================
        negation_patterns = [
            r'not\s+([abcde012345])',
            r'isn\'t\s+([abcde012345])',
            r'cannot\s+be\s+([abcde012345])',
            r'can\'t\s+be\s+([abcde012345])',
        ]

        negated_answers = set()
        for pattern in negation_patterns:
            for match in re.finditer(pattern, output_clean):
                negated_answers.add(match.group(1))

        if self.debug and negated_answers:
            print(f"🔍 Negated answers detected: {negated_answers}")

        # ================================================================
        # STRATEGY 4: First sentence with context filtering
        # ================================================================
        first_sentence_match = re.match(r'^([^.!?]+)', output_clean)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1).strip()

            # Context indicators that suggest it's NOT an answer
            false_positive_indicators = [
                'is a', 'as a', 'are a', 'was a', 'were a', 'be a',
                'like a', 'than a', 'such a', 'what a', 'have a',
                'has a', 'had a', 'with a', 'without a', 'for a',
                'to a', 'in a', 'on a', 'at a', 'of a', 'by a',
                'not a', 'just a', 'only a', 'quite a',
                ' a process', ' a method', ' a way', ' a system',
                ' a concept', ' a theory', ' a principle', ' a fact',
            ]

            # Find all letter/number matches in first sentence
            all_matches = list(re.finditer(r'\b([abcde012345])\b', first_sentence))

            for match in all_matches:
                found = match.group(1)

                # Skip if negated
                if found in negated_answers:
                    continue

                # Check context around the match
                start_pos = match.start()
                context_before = first_sentence[max(0, start_pos - 20):start_pos]

                # Skip if it's a false positive (e.g., "is a process")
                if any(phrase in context_before for phrase in false_positive_indicators):
                    continue

                # Check context after
                end_pos = match.end()
                if end_pos < len(first_sentence):
                    context_after = first_sentence[end_pos:min(len(first_sentence), end_pos + 20)]

                    # Skip if followed by alphanumeric (part of a word)
                    if context_after and context_after[0].isalnum():
                        continue

                    # Skip if followed by noun indicators
                    if any(noun in context_after for noun in [' process', ' method', ' way', ' system']):
                        continue

                # This looks like a valid answer!
                normalized = normalize_to_gt_format(found)
                strategy = "first_sentence_filtered"
                self._track_strategy(strategy)

                if self.debug:
                    print(f"✓ Strategy: {strategy} - Found: '{found}' → Normalized: '{normalized}'")
                    print(f"  Context: '{context_before}[{found}]{context_after[:10]}'")

                return (normalized == gt_str, found)

        # ================================================================
        # STRATEGY 5: Last occurrence in final 30% of output
        # ================================================================
        search_start = int(len(output_clean) * 0.7)
        search_region = output_clean[search_start:]

        all_matches = re.findall(r'\b([abcde012345])\b', search_region)
        if all_matches:
            # Take last match
            extracted = all_matches[-1]

            # Skip if negated
            if extracted not in negated_answers:
                normalized = normalize_to_gt_format(extracted)
                strategy = "last_occurrence_final_region"
                self._track_strategy(strategy)

                if self.debug:
                    print(f"✓ Strategy: {strategy} - Found: '{extracted}' → Normalized: '{normalized}'")
                    print(f"  All matches in final 30%: {all_matches}")

                return (normalized == gt_str, extracted)

        # ================================================================
        # STRATEGY 6: First occurrence (fallback)
        # ================================================================
        first_match = re.search(r'\b([abcde012345])\b', output_clean)
        if first_match:
            extracted = first_match.group(1)

            if extracted not in negated_answers:
                normalized = normalize_to_gt_format(extracted)
                strategy = "first_occurrence_fallback"
                self._track_strategy(strategy)

                if self.debug:
                    print(f"⚠️ Strategy: {strategy} (fallback) - Found: '{extracted}' → Normalized: '{normalized}'")

                return (normalized == gt_str, extracted)

        # Failed to extract
        if self.debug:
            print(f"✗ No multiple choice answer found in output")

        return (False, None)

    def _code_match(self, output: str, ground_truth: Any, tolerance: float) -> Tuple[bool, str]:
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', output, re.DOTALL)
        extracted_code = code_blocks[0].strip() if code_blocks else output.strip()

        try:
            ast.parse(extracted_code)
            has_valid_syntax = True
        except:
            has_valid_syntax = False

        is_exact_match = extracted_code == str(ground_truth).strip()
        return (is_exact_match or has_valid_syntax, extracted_code)


"""
Cross-Model Cognitive Archaeology: Analyzing Reasoning Divergence Across LLMs

This framework feeds identical prompts to multiple models and captures their
complete reasoning traces, building a "decision tree of thought" showing where
models diverge in their reasoning approaches.

Key Features:
1. Consistent prompting across all models
2. Complete reasoning trace extraction (step-by-step thinking)
3. Answer extraction tailored to task type
4. Divergence point detection
5. Comprehensive logging for future ablation studies
"""

import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelResponse:
    """Single model's response to one prompt"""
    model_name: str
    sample_id: int
    question: str
    ground_truth: Any
    task_type: str
    dataset_name: str

    # Generation details
    prompt_used: str
    full_generation: str
    reasoning_trace: List[str]  # Step-by-step reasoning extracted
    extracted_answer: Any
    is_correct: bool

    # Metadata
    generation_time: float
    num_reasoning_steps: int
    timestamp: str

    # Token-level details
    input_tokens: int
    output_tokens: int


@dataclass
class DivergencePoint:
    """A point where models' reasoning paths split"""
    sample_id: int
    question: str
    step_number: int
    step_description: str

    # Which models took which path
    path_a_models: List[str]
    path_b_models: List[str]
    path_a_reasoning: str
    path_b_reasoning: str

    # Downstream effects
    path_a_correct_count: int
    path_b_correct_count: int
    divergence_significance: float  # How different the paths are


# ============================================================================
# ANSWER VALIDATORS (From your emergence code)
# ============================================================================

class MathValidator:
    """Wrapper around RobustValidator for backward compatibility"""

    def __init__(self, debug=False):
        self.debug = debug
        # Import the robust validator
        self.validator = Validator(debug=debug)

    def extract_answer(self, text: str, ground_truth: Optional[float] = None) -> Optional[float]:
        """Extract numerical answer using robust validator

        Args:
            text: Model output text
            ground_truth: Optional ground truth for better filtering (prevents false positives)

        Returns:
            Extracted numeric value or None
        """
        # Use a high dummy value that won't interfere with extraction
        # The robust validator uses ground_truth for sanity checks
        dummy_gt = ground_truth if ground_truth is not None else 999999

        is_correct_dummy, extracted = self.validator._numeric_match(
            output=text,
            ground_truth=dummy_gt,
            tolerance=0.01
        )
        return extracted

    def validate(self, predicted: Optional[float], ground_truth: Any,
                 tolerance: float = 0.01) -> bool:
        """Check if prediction matches ground truth"""
        if predicted is None:
            return False

        if isinstance(ground_truth, (int, float)):
            return abs(predicted - ground_truth) <= tolerance
        else:
            return str(predicted) == str(ground_truth)
class ReasoningValidator:
    """Extract and validate reasoning tasks (BoolQ, etc.) - Using RobustValidator"""

    def __init__(self, debug=False):
        self.debug = debug
        self.validator = Validator(debug=debug)

    def extract_answer(self, text: str) -> Optional[bool]:
        """Extract yes/no answer using robust validator"""
        # Try yes first
        is_yes, extracted = self.validator._contains_match(
            output=text,
            ground_truth="yes",
            tolerance=0
        )

        if self.debug:
            print(f"[ReasoningValidator] Tried 'yes': is_yes={is_yes}, extracted={extracted}")

        # FIXED: Check extracted value, not is_yes boolean
        if extracted == "yes":
            return True

        # Try no
        is_no, extracted = self.validator._contains_match(
            output=text,
            ground_truth="no",
            tolerance=0
        )

        if self.debug:
            print(f"[ReasoningValidator] Tried 'no': is_no={is_no}, extracted={extracted}")

        # FIXED: Check extracted value, not is_no boolean
        if extracted == "no":
            return False

        if self.debug:
            print(f"[ReasoningValidator] FAILED to extract yes/no")

        return None

    def validate(self, predicted: Optional[bool], ground_truth: bool) -> bool:
        """Check if prediction matches ground truth"""
        return predicted == ground_truth


class MultipleChoiceValidator:
    """Extract and validate multiple choice answers - Using RobustValidator"""

    def __init__(self, debug=False):
        self.debug = debug
        self.validator = Validator(debug=debug)

    def extract_answer(self, text: str, task_name: str = None) -> Optional[str]:
        """Extract choice using robust validator"""
        # Try all possible choices
        for choice in ['a', 'b', 'c', 'd', 'e']:
            is_match, extracted = self.validator._multiple_choice_match(
                output=text,
                ground_truth=choice,
                tolerance=0,
                task_name=task_name
            )
            if extracted:
                return extracted
        return None

    def validate(self, predicted: Optional[str], ground_truth: Any) -> bool:
        """Check if prediction matches ground truth"""
        if predicted is None:
            return False

        gt_str = str(ground_truth).strip().lower()
        pred_str = str(predicted).strip().lower()

        # Direct match
        if gt_str == pred_str:
            return True

        # Try letter matching
        if len(gt_str) == 1 and len(pred_str) == 1:
            return gt_str == pred_str

        return False