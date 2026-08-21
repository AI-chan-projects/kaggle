# 1. 가장 기본 (Gemma+Gemma, seed 고정, 재현 가능)
python -m main --seed 42

# 2. Gemma(guesser) + Flan-T5(answerer), T5는 작으니 양자화 끔
python -m main --seed 42 \
  --guesser-model google/gemma-1.1-2b-it --guesser-family causal \
  --answerer-model google/flan-t5-base --answerer-family seq2seq --no-answerer-quantize

# 3. 둘 다 Flan-T5 (양자화 둘 다 끔)
python -m main --seed 42 \
  --guesser-model google/flan-t5-large --guesser-family seq2seq --no-guesser-quantize \
  --answerer-model google/flan-t5-base --answerer-family seq2seq --no-answerer-quantize

# 4. 프롬프트 variant 비교 (같은 seed로 baseline vs category_first_strict)
python -m main --seed 42 --variant baseline
python -m main --seed 42 --variant category_first_strict
python -m main --seed 42 --variant explicit_category_list

# 5. 턴 수 줄여서 빠르게 디버깅
python -m main --seed 42 --max-turns 5

# 6. Gemma(guesser, 양자화) + Gemma(answerer, 양자화 끔) — 같은 모델이지만 다른 설정
python -m main --seed 42 \
  --guesser-model google/gemma-1.1-2b-it --guesser-quantize \
  --answerer-model google/gemma-1.1-2b-it --no-answerer-quantize