import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

class ModelMerger:
    """모델 병합을 담당하는 클래스"""
    
    def __init__(self, base_model_path: str, cache_dir: str = "/data"):
        self.base_model_path = base_model_path
        self.cache_dir = cache_dir
        
    def merge_and_save(self, adapter_path: str, output_path: str):
        """어댑터를 병합하고 저장합니다"""
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map='auto',
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
        
        # 어댑터와 병합
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        
        # 병합 및 저장
        merged_model = model_with_adapter.merge_and_unload()
        merged_model.save_pretrained(output_path)
        
        return merged_model