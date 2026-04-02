
import re
from typing import List




# ============================================================================
# REASONING TRACE EXTRACTOR
# ============================================================================

class ReasoningTraceExtractor:
    """Extract step-by-step reasoning from model outputs"""

    def __init__(self, debug=False):
        self.debug = debug

    def extract_steps(self, text: str, task_type: str = "math") -> List[str]:
        """
        Extract reasoning steps from generated text.
        Returns list of reasoning steps.
        """
        steps = []

        # Strategy 1: Look for explicit step markers
        step_patterns = [
            r'Step\s+\d+[:\-]\s*(.+?)(?=Step\s+\d+|$)',
            r'\d+[\.:\)]\s*(.+?)(?=\d+[\.:\)]|$)',
            r'First[,\s]+(.+?)(?=Second|Next|Then|Finally|$)',
            r'(?:Next|Then)[,\s]+(.+?)(?=Next|Then|Finally|$)',
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                steps = [m.strip() for m in matches if m.strip()]
                if len(steps) >= 2:  # Found at least 2 steps
                    return steps

        # Strategy 2: Split by sentences and identify reasoning sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        reasoning_indicators = [
            'calculate', 'compute', 'find', 'determine', 'solve',
            'first', 'then', 'next', 'after', 'finally',
            'so', 'therefore', 'thus', 'hence',
            'multiply', 'divide', 'add', 'subtract',
            'let', "let's", 'we need', 'we can', 'we have'
        ]

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                steps.append(sentence)

        # Strategy 3: Fallback - just split into sentences
        if len(steps) < 2:
            steps = sentences[:10]  # Take first 10 sentences

        return steps