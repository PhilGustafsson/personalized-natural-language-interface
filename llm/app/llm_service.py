"""
LLM Service using vLLM for efficient text generation.
Specialized for generating domain-specific training prompts.
"""

from vllm import LLM, SamplingParams
from typing import List, Optional
import re


class LLMService:
    """Service for generating domain-specific prompts using vLLM."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = 2048,
        gpu_memory_utilization: float = 0.75
    ):
        """
        Initialize vLLM service.
        
        Args:
            model_name: HuggingFace model name or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
    
    def load_model(
        self,
        model_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None
    ):
        """
        Load the vLLM model.
        
        Args:
            model_name: Override default model name
            tensor_parallel_size: Override tensor parallel size
            max_model_len: Override max model length
            gpu_memory_utilization: Override GPU memory utilization
        """
        if model_name:
            self.model_name = model_name
        if tensor_parallel_size:
            self.tensor_parallel_size = tensor_parallel_size
        if max_model_len:
            self.max_model_len = max_model_len
        if gpu_memory_utilization:
            self.gpu_memory_utilization = gpu_memory_utilization
        
        print(f"Loading vLLM model: {self.model_name}")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            download_dir="/llm-app/models"
        )
        print(f"Model loaded successfully: {self.model_name}")
    
    def unload_model(self):
        """Unload model from memory to free VRAM."""
        if self.llm is not None:
            print(f"Unloading model: {self.model_name}")
            del self.llm
            self.llm = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded and VRAM freed")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 0.9,
        frequency_penalty: float = 0.3
    ) -> str:
        """
        Generate text using chat format.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query/request
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for token repetition
            
        Returns:
            Generated text string
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format as chat messages (Qwen format)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Create formatted prompt (simplified - vLLM handles chat formatting)
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        return generated_text
    
    def generate_domain_prompts(
        self,
        domain: str,
        num_prompts: int = 20,
        language: str = "sv",
        difficulty: str = "intermediate",
        sentence_length: str = "medium",
        include_technical_terms: bool = True,
        style: str = "conversational",
        temperature: float = 0.8,
        max_tokens: int = 150
    ) -> List[str]:
        """
        Generate domain-specific training prompts.
        
        Args:
            domain: Domain/topic (e.g., "AI", "medicine", "finance")
            num_prompts: Number of prompts to generate
            language: Target language ("sv" for Swedish, "en" for English)
            difficulty: Difficulty level
            sentence_length: Desired length
            include_technical_terms: Whether to include domain terminology
            style: Writing style
            temperature: Sampling temperature
            max_tokens: Max tokens per generation
            
        Returns:
            List of generated prompt strings
        """
        # Build system prompt
        language_map = {
            "sv": "Swedish",
            "en": "English"
        }
        
        lang_full = language_map.get(language, "Swedish")
        
        system_prompt = f"""You are an expert prompt generator for speech training data. 
Your task is to generate natural, readable sentences in {lang_full} for the domain of {domain}.

Requirements:
- Language: {lang_full}
- Difficulty: {difficulty}
- Sentence length: {sentence_length} (short=5-10 words, medium=10-20 words, long=20-35 words)
- Style: {style}
- Technical terms: {'Include domain-specific terminology' if include_technical_terms else 'Use simple language'}

Guidelines:
- Create sentences that are natural to speak aloud
- Vary sentence structure and vocabulary
- For Swedish: Mix in relevant English technical terms naturally (code-mixing is common in Swedish tech/academic contexts)
- Cover different aspects of the domain
- Make sentences clear and unambiguous
- Avoid overly complex nested clauses

Generate ONLY the sentences, one per line. No numbering, no extra formatting."""

        user_prompt = f"""Generate {num_prompts} training sentences about {domain} following the requirements above.

Examples for {domain}:
- Mix declarative statements, questions, and explanations
- Include both general concepts and specific terminology
- Vary complexity within the {difficulty} level

Sentences:"""

        generated_text = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens * num_prompts,  # Allocate tokens for all prompts
            frequency_penalty=0.5  # Encourage diversity
        )
        
        # Parse generated text into individual prompts
        prompts = self._parse_prompts(generated_text, num_prompts)
        
        return prompts
    
    def _parse_prompts(self, generated_text: str, expected_count: int) -> List[str]:
        """
        Parse generated text into individual prompts.
        
        Args:
            generated_text: Raw generated text
            expected_count: Expected number of prompts
            
        Returns:
            List of cleaned prompt strings
        """
        # Split by newlines and filter
        lines = generated_text.strip().split('\n')
        
        prompts = []
        for line in lines:
            # Remove numbering, bullets, etc.
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Filter out empty lines and very short lines
            if cleaned and len(cleaned.split()) >= 3:
                prompts.append(cleaned)
        
        # If we got fewer prompts than expected, fill with variations
        if len(prompts) < expected_count:
            print(f"Warning: Generated {len(prompts)} prompts, expected {expected_count}")
        
        # Return up to expected count
        return prompts[:expected_count]
