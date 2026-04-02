


# ============================================================================
# PROMPT FORMATTER
# ============================================================================

class PromptFormatter:
    """Create consistent prompts across all models"""

    @staticmethod
    def format(question: str, task_type: str = "math") -> str:
        """
        Format question with explicit instructions for step-by-step reasoning.

        CRITICAL: This prompt MUST be identical for ALL models to ensure
        fair comparison of reasoning approaches.
        """

        if task_type == "math":
            return f"""Problem: {question}

Solution:"""

        elif task_type == "reasoning":
          return f"""Read the passage carefully and answer the question.

      Passage: {question.split('Question:')[0].replace('Passage:', '').strip()}

      Question: {question.split('Question:')[1].strip()}

      Think carefully about what the passage says. Answer with only "Yes" or "No".

      Answer:"""

        elif task_type == "commonsense":
            return f"""Question: {question}

        Answer:"""

        else:
            return question
