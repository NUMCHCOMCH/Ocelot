import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """공통 로깅 유틸리티 클래스"""
    
    _instances = {}
    
    def __new__(cls, name: str = "jobtalks_ocelot"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str = "jobtalks_ocelot"):
        if hasattr(self, '_initialized'):
            return
        
        self.name = name
        self.logger = logging.getLogger(name)
        self._initialized = True
        
        # 기본 설정이 아직 안되어 있으면 설정
        if not self.logger.handlers:
            self._setup_default_logger()
    
    def _setup_default_logger(self):
        """기본 로거 설정"""
        self.logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def setup_file_logging(self, log_dir: Optional[str] = None, log_level: str = "INFO"):
        """파일 로깅 설정 추가"""
        if log_dir is None:
            log_dir = "logs"
        
        # 로그 디렉토리 생성
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # 파일 핸들러 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 파일용 포맷터 (더 상세한 정보 포함)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        
        return str(log_file)
    
    def setup_training_logging(self, output_dir: str):
        """훈련용 특별 로깅 설정"""
        log_file = self.setup_file_logging(f"{output_dir}/logs", "DEBUG")
        self.info(f"Training logs will be saved to: {log_file}")
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def log_training_step(self, step: int, loss: float, lr: float = None):
        """훈련 스텝 로깅"""
        msg = f"Step {step}: Loss = {loss:.4f}"
        if lr is not None:
            msg += f", LR = {lr:.2e}"
        self.info(msg)
    
    def log_evaluation_result(self, metric_name: str, value: float):
        """평가 결과 로깅"""
        self.info(f"Evaluation - {metric_name}: {value:.4f}")
    
    def log_model_info(self, model_name: str, num_parameters: int):
        """모델 정보 로깅"""
        self.info(f"Model: {model_name}, Parameters: {num_parameters:,}")
    
    def log_dataset_info(self, split: str, size: int):
        """데이터셋 정보 로깅"""
        self.info(f"Dataset {split}: {size:,} samples")


# 전역 로거 인스턴스
def get_logger(name: str = "jobtalks_ocelot") -> Logger:
    """전역 로거 인스턴스 반환"""
    return Logger(name)


# 편의 함수들
def setup_logging(log_dir: Optional[str] = None, log_level: str = "INFO") -> Logger:
    """빠른 로깅 설정"""
    logger = get_logger()
    if log_dir:
        logger.setup_file_logging(log_dir, log_level)
    return logger


def log_function_call(func_name: str, **kwargs):
    """함수 호출 로깅 데코레이터용 유틸리티"""
    logger = get_logger()
    args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({args_str})")


# 훈련용 편의 함수들
def log_training_start(model_name: str, dataset_size: int, epochs: int):
    """훈련 시작 로깅"""
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("TRAINING STARTED")
    logger.info("=" * 50)
    logger.log_model_info(model_name, 0)  # 파라미터 수는 별도로 설정
    logger.log_dataset_info("train", dataset_size)
    logger.info(f"Training epochs: {epochs}")


def log_training_end(total_time: float):
    """훈련 완료 로깅"""
    logger = get_logger()
    logger.info("=" * 50)
    logger.info(f"TRAINING COMPLETED in {total_time:.2f} seconds")
    logger.info("=" * 50)


def log_inference_start(prompt: str):
    """추론 시작 로깅"""
    logger = get_logger()
    logger.info("=" * 30)
    logger.info("INFERENCE STARTED")
    logger.info("=" * 30)
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")


def log_inference_result(result: str, generation_time: float):
    """추론 결과 로깅"""
    logger = get_logger()
    logger.info(f"Generation time: {generation_time:.2f} seconds")
    logger.info(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    logger.info("=" * 30)
