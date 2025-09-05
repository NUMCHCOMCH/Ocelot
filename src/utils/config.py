from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ModelConfig:
    """모델 설정을 관리하는 클래스"""
    base_model: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    cache_dir: str = "/data"
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_k: int = 50
    top_p: float = 0.95

@dataclass
class LoraConfig:
    """LoRA 설정을 관리하는 클래스"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "o_proj", "k_proj", "v_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """훈련 설정을 관리하는 클래스"""
    output_dir: str = "/data/model_storage"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 2e-6
    fp16: bool = True
    logging_steps: int = 25
    push_to_hub: bool = False
    report_to: str = 'none'
    max_seq_length: int = 4096

@dataclass
class QuantizationConfig:
    """양자화 설정을 관리하는 클래스"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"