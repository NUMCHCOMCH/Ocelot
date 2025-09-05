from huggingface_hub import notebook_login
from src.utils.config import ModelConfig, LoraConfig, QuantizationConfig, TrainingConfig
from src.utils.log import get_logger, setup_logging, log_training_start, log_training_end
from src.datasets.processing.data_pipeline import DatasetManager, PromptFormatter
from src.model.model_manager import ModelManager
from src.inference.model_inference import InferenceEngine
from src.train.eeve_sft import TrainerManager
from src.train.eeve_merge import ModelMerger

class KoreanResumeFineTuner:
    """전체 파인튜닝 프로세스를 관리하는 메인 클래스"""
    
    def __init__(self):
        # 로깅 설정
        self.logger = setup_logging("logs", "INFO")
        
        # 설정 초기화
        self.model_config = ModelConfig()
        self.lora_config = LoraConfig()
        self.training_config = TrainingConfig()
        self.quant_config = QuantizationConfig()
        
        # 컴포넌트 초기화
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager(self.model_config)
        self.prompt_formatter = None
        self.inference_engine = None
        self.trainer_manager = None
        
    def setup(self):
        """초기 설정을 수행합니다"""
        self.logger.info("초기 설정을 시작합니다...")
        
        # HuggingFace 로그인
        self.logger.info("HuggingFace 로그인 중...")
        notebook_login()
        
        # 데이터셋 로드
        self.logger.info("데이터셋을 로드합니다...")
        self.dataset_manager.load_dataset()
        
        # 토크나이저 로드
        self.logger.info("토크나이저를 로드합니다...")
        tokenizer = self.model_manager.load_tokenizer()
        self.prompt_formatter = PromptFormatter(tokenizer)
        
        self.logger.info("초기 설정이 완료되었습니다.")
        
    def run_inference_test(self):
        """추론 테스트를 실행합니다"""
        # 기본 모델로 추론 테스트
        model = self.model_manager.load_model()
        self.inference_engine = InferenceEngine(model, self.model_manager.tokenizer, self.model_config)
        
        # 샘플 추론
        sample_instruction = self.dataset_manager.get_sample_instruction(0)
        messages = self.prompt_formatter.format_chat_messages(sample_instruction)
        prompt = self.prompt_formatter.apply_chat_template(messages)
        
        print("=== 추론 결과 ===")
        result = self.inference_engine.generate(prompt)
        print(result)
        
    def run_training(self, adapter_name: str = "lora_adapter-v5"):
        """훈련을 실행합니다"""
        # 양자화된 모델 로드
        quant_config = self.model_manager.create_quantization_config(self.quant_config)
        model = self.model_manager.load_model(quant_config)
        
        # LoRA 설정 생성
        lora_config = self.model_manager.create_lora_config(self.lora_config)
        
        # 훈련 데이터 준비
        train_data = self.dataset_manager.get_train_data()
        
        # 트레이너 생성 및 훈련
        self.trainer_manager = TrainerManager(
            model=model,
            train_dataset=train_data,
            tokenizer=self.model_manager.tokenizer,
            training_config=self.training_config,
            lora_config=lora_config,
            formatting_func=self.prompt_formatter.generate_training_prompt
        )
        
        print("=== 훈련 시작 ===")
        self.trainer_manager.train()
        
        print("=== 어댑터 저장 ===")
        self.trainer_manager.save_adapter(adapter_name)
        
        return adapter_name
        
    def merge_and_save_model(self, adapter_path: str, output_path: str = "/data/solar-ko-resume"):
        """모델을 병합하고 저장합니다"""
        merger = ModelMerger(self.model_config.base_model, self.model_config.cache_dir)
        
        print("=== 모델 병합 및 저장 ===")
        merged_model = merger.merge_and_save(adapter_path, output_path)
        print(f"병합된 모델이 {output_path}에 저장되었습니다.")
        
        return merged_model
        
    def run_full_pipeline(self):
        """전체 파이프라인을 실행합니다"""
        print("=== 설정 초기화 ===")
        self.setup()
        
        print("=== 추론 테스트 ===")
        self.run_inference_test()
        
        print("=== 파인튜닝 훈련 ===")
        adapter_name = self.run_training()
        
        print("=== 모델 병합 ===")
        self.merge_and_save_model(adapter_name)
        
        print("=== 파이프라인 완료 ===")


# 사용 예시
if __name__ == "__main__":
    # 전체 파이프라인 실행
    finetuner = KoreanResumeFineTuner()
    finetuner.run_full_pipeline()