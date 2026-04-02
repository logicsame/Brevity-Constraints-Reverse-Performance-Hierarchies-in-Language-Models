from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manage model loading and generation"""

    def __init__(self, model_name: str, device: str = "cuda", debug: bool = False):
        self.model_name = model_name
        self.device = device
        self.debug = debug

        print(f"🔧 Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        print(f"✅ Model loaded: {model_name}")

    def detect_repetition(self, text: str, threshold: int = 3) -> bool:
        """Detect if model is stuck repeating"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < threshold:
            return False

        # Check last N sentences
        last_n = sentences[-threshold:]
        if len(set(last_n)) == 1:  # All identical
            return True

        # Check for specific phrase repetition
        repetitive_phrases = [
            'The question asked',
            'The answer is',
            'Therefore',
        ]

        for phrase in repetitive_phrases:
            if text.count(phrase) > 3:
                return True

        return False

    def generate(self, prompt: str, max_new_tokens: int = 300) -> Tuple[str, float, int, int]:
        """Generate with early stopping when answer found"""
        import time

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_tokens = inputs['input_ids'].shape[1]

        start_time = time.time()



        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=1.0,
                top_p=1.0,
            )

        generation_time = time.time() - start_time

        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[1] - input_tokens

        # Check for repetition
        generation_only = full_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        if self.detect_repetition(generation_only):
            if self.debug:
                print(f"⚠️ REPETITION DETECTED - Output may be unreliable")

        return full_text, generation_time, input_tokens, output_tokens

    def cleanup(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            print(f"🗑️ Cleaned up {self.model_name}")