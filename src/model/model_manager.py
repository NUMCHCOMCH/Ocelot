import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig as PeftLoraConfig, PeftModel
from typing import Optional
from config.config import ModelConfig, LoraConfig, QuantizationConfig

class ModelManager:
    """모델 로드 및 관리를 담당하는 클래스"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.streamer = None
        
    def load_tokenizer(self) -> AutoTokenizer:
        """토크나이저를 로드합니다"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            add_special_tokens=True,
            cache_dir=self.config.cache_dir
        )
        self.streamer = TextStreamer(self.tokenizer)
        return self.tokenizer
    
    def load_model(self, quantization_config: Optional[BitsAndBytesConfig] = None) -> AutoModelForCausalLM:
        """모델을 로드합니다"""
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                device_map='auto',
                quantization_config=quantization_config,
                cache_dir=self.config.cache_dir
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                device_map='auto',
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16
            )
        return self.model
    
    def create_quantization_config(self, quant_config: QuantizationConfig) -> BitsAndBytesConfig:
        """양자화 설정을 생성합니다"""
        compute_dtype = getattr(torch, quant_config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=quant_config.load_in_4bit,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )
    
    def create_lora_config(self, lora_config: LoraConfig) -> PeftLoraConfig:
        """LoRA 설정을 생성합니다"""
        return PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            task_type=lora_config.task_type,
        )