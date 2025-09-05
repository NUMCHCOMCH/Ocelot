from transformers import pipeline
from typing import List, Dict, Any
from config.config import ModelConfig

class InferenceEngine:
    """추론을 담당하는 클래스"""
    
    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.pipeline = None
        
    def create_pipeline(self):
        """텍스트 생성 파이프라인을 생성합니다"""
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
        )
        return self.pipeline
    
    def generate(self, prompt: str, do_sample: bool = True) -> str:
        """텍스트를 생성합니다"""
        if self.pipeline is None:
            self.create_pipeline()
            
        outputs = self.pipeline(
            prompt,
            do_sample=do_sample,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            add_special_tokens=True
        )
        return outputs[0]["generated_text"][len(prompt):]