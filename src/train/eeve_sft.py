from transformers import TrainingArguments
from config.config import TrainingConfig
from trl import SFTTrainer
from typing import Callable

class TrainerManager:
    """훈련을 담당하는 클래스"""
    
    def __init__(self, model, train_dataset, tokenizer, 
                 training_config: TrainingConfig, 
                 lora_config, 
                 formatting_func: Callable):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.lora_config = lora_config
        self.formatting_func = formatting_func
        self.trainer = None
        
    def create_training_arguments(self) -> TrainingArguments:
        """훈련 인수를 생성합니다"""
        return TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            max_steps=self.training_config.max_steps,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            optim=self.training_config.optim,
            learning_rate=self.training_config.learning_rate,
            fp16=self.training_config.fp16,
            logging_steps=self.training_config.logging_steps,
            push_to_hub=self.training_config.push_to_hub,
            report_to=self.training_config.report_to,
        )
    
    def create_trainer(self) -> SFTTrainer:
        """SFT 트레이너를 생성합니다"""
        training_args = self.create_training_arguments()
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            max_seq_length=self.training_config.max_seq_length,
            args=training_args,
            peft_config=self.lora_config,
            formatting_func=self.formatting_func,
        )
        return self.trainer
    
    def train(self):
        """훈련을 시작합니다"""
        if self.trainer is None:
            self.create_trainer()
        return self.trainer.train()
    
    def save_adapter(self, adapter_path: str):
        """어댑터를 저장합니다"""
        if self.trainer is None:
            raise ValueError("훈련을 먼저 실행해야 합니다.")
        self.trainer.model.save_pretrained(adapter_path)