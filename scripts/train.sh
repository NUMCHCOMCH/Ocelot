#!/bin/bash

# =========================================================
# Axolotl 학습 스크립트
#
# 이 스크립트는 Train 폴더의 YAML 설정 파일을 사용하여
# Axolotl 라이브러리를 통해 LLM 학습을 시작합니다.
# =========================================================

# --- 설정 변수 ---
# 루트 디렉토리 경로 (현재 스크립트가 실행되는 위치에 따라 조정)
TRAIN_DIR="."

# Axolotl 설정 YAML 파일 이름 (루트 디렉토리에 있어야 함)
# 예: config.yaml, my_training_config.yaml 등
CONFIG_FILE="solar.yml" 

# Axolotl 실행 스크립트 경로 (일반적으로 'axolotl' 또는 'python -m axolotl.cli.train')
# 시스템 환경에 따라 'axolotl' 명령어 바로 사용 가능하거나,
# Axolotl 설치 경로에 따라 달라질 수 있습니다.
# 예를 들어, 가상 환경을 활성화하거나 python -m을 사용할 수 있습니다.
AXOLOTL_EXEC="accelerate launch -m axolotl.cli.train" # 권장: accelerate launch 사용

# --- 스크립트 시작 ---

echo "========================================================="
echo "Axolotl LLM 학습 시작"
echo "========================================================="

# 1. 루트 디렉토리 존재 여부 확인
if [ ! -d "$TRAIN_DIR" ]; then
    echo "오류: '${TRAIN_DIR}' 폴더를 찾을 수 없습니다."
    echo "스크립트가 실행되는 위치 또는 TRAIN_DIR 변수를 확인해주세요."
    exit 1
fi

# 2. 설정 YAML 파일 존재 여부 확인
FULL_CONFIG_PATH="${TRAIN_DIR}/${CONFIG_FILE}"
if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "오류: 설정 파일 '${FULL_CONFIG_PATH}'을(를) 찾을 수 없습니다."
    echo "CONFIG_FILE 변수를 확인하거나, 루트 디렉토리에 해당 파일이 있는지 확인해주세요."
    exit 1
fi

echo "Axolotl 설정 파일: ${FULL_CONFIG_PATH}"
echo "Axolotl 실행 명령어: ${AXOLOTL_EXEC}"

# 3. Axolotl 실행 (accelerate launch를 통해 분산 학습 지원)
# `accelerate launch`는 Axolotl의 분산 학습을 위해 권장되는 방법입니다.
# 만약 단일 GPU에서만 실행하거나, Accelerate가 필요 없는 경우 단순히 `axolotl ${FULL_CONFIG_PATH}`를 사용할 수도 있습니다.
# 하지만 대부분의 LLM 학습은 Accelerate가 필수적입니다.
echo "Axolotl 학습 실행 중..."
${AXOLOTL_EXEC} ${FULL_CONFIG_PATH}

# 4. 종료 메시지
if [ $? -eq 0 ]; then
    echo "========================================================="
    echo "Axolotl 학습이 성공적으로 완료되었습니다."
    echo "========================================================="
else
    echo "========================================================="
    echo "오류: Axolotl 학습 중 문제가 발생했습니다."
    echo "위의 로그를 확인하여 문제를 해결해주세요."
    echo "========================================================="
    exit 1
fi
