#!/usr/bin/env bash
# scripts/eval_ko_model.sh
# Korean LLM 평가 자동화 스크립트
# - Beomi/ko-lm-evaluation-harness 기준
# - 모델/디바이스/캐시/가상환경 옵션 제공

set -euo pipefail

########## [사용자가 필요시 수정할 변수] ########################################
MODEL_NAME="${MODEL_NAME:-cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0}"  # 평가할 HF 모델 이름 또는 로컬 경로
REPO_URL="${REPO_URL:-https://github.com/Beomi/ko-lm-evaluation-harness}"
REPO_DIR="${REPO_DIR:-ko-lm-evaluation-harness}"

# Python/환경 관련
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-true}"                  # true|false
VENV_DIR="${VENV_DIR:-.venv-ko-lm-eval}"

# 실행 옵션
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # 여러 GPU면 "0,1" 등
DTYPE="${DTYPE:-float16}"                         # float16|bfloat16|float32
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"    # 일부 모델은 필요
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}"
TRANSFORMERS_CACHE_DIR="${TRANSFORMERS_CACHE_DIR:-$HF_HOME_DIR/transformers}"

# 결과 수집
RESULTS_ROOT="${RESULTS_ROOT:-results}"
###############################################################################

echo ">>> Evaluating the model: ${MODEL_NAME}"

# 0) 준비: 디렉토리/의존성
mkdir -p "${RESULTS_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES

# 1) 리포지토리 클론
if [ ! -d "${REPO_DIR}" ]; then
  echo ">>> Cloning repo: ${REPO_URL}"
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  echo ">>> Repo already exists: ${REPO_DIR} (pull latest)"
  pushd "${REPO_DIR}" >/dev/null
  git pull --ff-only || true
  popd >/dev/null
fi

# 2) 가상환경
if [ "${USE_VENV}" = "true" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    echo ">>> Creating venv: ${VENV_DIR}"
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PY="${VENV_DIR}/bin/python"
  PIP="${VENV_DIR}/bin/pip"
else
  PY="${PYTHON_BIN}"
  PIP="$(command -v pip || echo pip)"
fi

# 3) 의존성 설치
echo ">>> Installing requirements"
pushd "${REPO_DIR}" >/dev/null
${PIP} install --upgrade pip wheel
if [ -f requirements.txt ]; then
  ${PIP} install -r requirements.txt
else
  # 백업 플로우: lm-eval-harness + transformers 등
  ${PIP} install "lm-eval==0.4.2" "transformers>=4.41.0" accelerate datasets sentencepiece
fi

# 4) 실행 권한/환경 변수 구성
chmod +x ./run_all.sh || true

# ko-lm-evaluation-harness는 내부에서 ‘lm_eval … --model hf --model_args …’를 호출합니다.
# 필요 시 MODEL_ARGS를 오버라이드할 수 있게 제공
MODEL_ARGS="pretrained=${MODEL_NAME},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE}"

# 5) 평가 실행
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
OUT_DIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo ">>> Starting evaluation"
echo "     MODEL_NAME = ${MODEL_NAME}"
echo "     MODEL_ARGS = ${MODEL_ARGS}"
echo "     CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "     DTYPE = ${DTYPE}"

# 일부 run_all.sh가 환경변수 인식을 하도록 다음을 export
export MODEL_NAME
export MODEL_ARGS
export HF_HOME
export TRANSFORMERS_CACHE
export CUDA_VISIBLE_DEVICES

set +e
./run_all.sh
EC=$?
set -e

# 6) 결과 수집
# 관례적으로 results/ 또는 output/ 아래에 json/csv/log가 생깁니다.
# 존재하는 디렉토리/파일을 OUT_DIR로 모읍니다.
echo ">>> Collecting outputs"
for d in results result output outputs eval_results; do
  if [ -d "${d}" ]; then
    # 중첩 결과까지 안전 복사
    rsync -a "${d}/" "${OUT_DIR}/" || true
  fi
done
# 루트에 떨어진 .json/.csv도 수집
find . -maxdepth 1 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.log" \) -exec cp {} "${OUT_DIR}/" \; || true

popd >/dev/null

# 7) 요약 출력
echo
echo "==============================================="
echo "Evaluation finished (exit code: ${EC})"
echo "Results saved to: ${REPO_DIR}/${OUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "DType: ${DTYPE}"
echo "==============================================="

# 8) 실패 시 힌트
if [ "${EC}" -ne 0 ]; then
  echo "⚠️  run_all.sh returned non-zero (${EC})."
  echo "   - 모델이 trust_remote_code를 요구할 수 있습니다: TRUST_REMOTE_CODE=true"
  echo "   - 메모리 부족 시: CUDA_VISIBLE_DEVICES 조정, DTYPE=bfloat16/float16 확인"
  echo "   - CPU만 사용 시 매우 느릴 수 있습니다."
  exit "${EC}"
fi
