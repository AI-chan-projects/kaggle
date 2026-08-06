"""
llm_20_questions 로컬 실행용 main.py
- 레퍼런스 파일(llm_20_questions.py)의 agents/keyword_guessed 등을 그대로 재사용
- call_llm만 T5 -> Gemma-1.1-2b-it (4bit 양자화)로 교체
- kaggle_environments의 4-agent interpreter는 제출용이라 로컬 테스트엔 과함 ->
  guesser/answerer 한 쌍으로 직접 도는 단순 루프로 대체
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 레퍼런스 파일이 llm_20_questions.py 로 저장되어 있다고 가정
# (파일명이 다르면 이 import만 바꿔주면 됨)
import llm_20_questions as ref
from llm_20_questions import GUESSER, ANSWERER, ASK, GUESS, keyword_guessed

# ---------------- 프롬프트 실험용 인터페이스 ----------------
# call_llm을 몽키패치했던 것과 같은 원리: guesser_agent/answerer_agent 내부에서
# 이름으로 참조하는 ref 모듈의 전역 상수(QUESTION_STRATEGY_EARLY 등)를
# 실행 전에 덮어쓰면, llm_20_questions.py 코드는 한 줄도 안 건드리고
# 프롬프트만 바꿔가며 테스트할 수 있다.
#
# variant 하나 추가하고 싶으면 아래 딕셔너리에 항목만 추가하면 됨.
PROMPT_VARIANTS = {
    "baseline": {},  # llm_20_questions.py에 정의된 기본 프롬프트 그대로 사용

    "category_first_strict": {
        "QUESTION_STRATEGY_EARLY": (
            "Strategy: this is the very first question. You MUST ask about "
            "ONLY ONE of these categories: person, place, animal, or object — "
            "never list more than one option in a single question. Your "
            "question must be answerable with a strict yes or no. "
            "Good: 'Is it a place?' Bad: 'Is it a person, a place, an animal, "
            "or an object?' Ask about whichever single category you think is "
            "most likely."
        ),
    },

    "explicit_category_list": {
        "QUESTION_STRATEGY_EARLY": (
            "Strategy: this is the very first question. The keyword belongs "
            "to one of these categories: person, place (city/country/"
            "landmark), animal, or object. Ask a single yes/no question "
            "testing ONE of these categories alone — never combine multiple "
            "categories with 'or' in one question, since that cannot be "
            "answered with yes/no. "
            "Good: 'Is it a place?' Bad: 'Is it a person or a place?'"
        ),
        "QUESTION_STRATEGY_LATE": (
            "Strategy: you already know the broad category from the yes/no "
            "answers above — do NOT ask about the broad category again. "
            "Look at the Q&A history and ask a NEW question that cuts the "
            "remaining possibilities roughly in half. Never ask a question "
            "whose answer you can already infer from the history above."
        ),
    },
}


def apply_prompt_variant(name: str) -> None:
    """ref(llm_20_questions) 모듈의 프롬프트 상수를 variant 값으로 오버라이드."""
    if name not in PROMPT_VARIANTS:
        raise ValueError(f"알 수 없는 variant: {name} (선택 가능: {list(PROMPT_VARIANTS)})")
    overrides = PROMPT_VARIANTS[name]
    for attr, value in overrides.items():
        if not hasattr(ref, attr):
            raise AttributeError(f"llm_20_questions.py에 '{attr}' 상수가 없습니다.")
        setattr(ref, attr, value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default="baseline", choices=list(PROMPT_VARIANTS),
        help="테스트할 프롬프트 variant 이름",
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="지정하면 torch.manual_seed로 고정 (variant 간 비교 재현성용). 안 주면 매 실행 랜덤.",
    )
    return parser.parse_args()


ARGS = parse_args()

# ---------------- 로깅 설정 ----------------
# 파일명은 실행 시작 시각 + variant 이름 기준. 콘솔 출력 구조는 print와 동일하게 유지.
START_TIME = datetime.now()
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"{START_TIME:%Y%m%d_%H%M%S}_{ARGS.variant}.log"

logger = logging.getLogger("llm20q")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _formatter = logging.Formatter("%(message)s")
    _file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_stream_handler)

# ---------------- Gemma 설정 ----------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

MODEL_NAME = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
)


def call_llm(prompt: str) -> str:
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # BatchEncoding(dict)으로 고정 반환 -> generate(**inputs)로 사용
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # 프롬프트 끝의 cue로 ask/guess를 구분해서 다른 생성 파라미터 적용.
    # - 질문(ask): 다양성이 필요 -> temperature 조금 높게, greedy가 "Paris" 류로
    #   수렴하는 걸 완화
    # - 추측(guess): 창의성 불필요, 정확한 단어 하나가 필요 -> temperature 낮게.
    #   지난 실행에서 guess에 샘플링을 그대로 쓰니 "Statue podrobly." 같은
    #   의미 없는 문자열이 나왔음.
    if prompt.rstrip().endswith("Guess:"):
        gen_kwargs = dict(do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.1)
    else:
        gen_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15)

    output_ids = model.generate(**inputs, max_new_tokens=64, **gen_kwargs)
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# 레퍼런스 모듈의 call_llm을 Gemma 버전으로 교체 (guesser_agent/answerer_agent 내부에서 사용됨)
ref.call_llm = call_llm


# ---------------- 게임 루프 ----------------
def make_obs(role, turn_type, questions, answers, guesses):
    return SimpleNamespace(
        role=role,
        turnType=turn_type,
        questions=questions,
        answers=answers,
        guesses=guesses,
        keyword=ref.keyword,
        category=ref.category,
    )


def run_game(max_turns: int = 20, verbose: bool = True):
    questions, answers, guesses = [], [], []
    guessed = False
    score = 0

    for turn in range(1, max_turns + 1):
        # 1) Guesser가 질문
        ask_obs = make_obs(GUESSER, ASK, questions, answers, guesses)
        question = ref.guesser_agent(ask_obs).strip()
        questions.append(question)

        # 2) Answerer가 답변
        ans_obs = make_obs(ANSWERER, "answer", questions, answers, guesses)
        raw = ref.answerer_agent(ans_obs).strip().lower()
        answer = "yes" if "yes" in raw else "no" if "no" in raw else "maybe"
        answers.append(answer)

        # 3) Guesser가 추측
        guess_obs = make_obs(GUESSER, GUESS, questions, answers, guesses)
        guess = ref.guesser_agent(guess_obs).strip()
        guesses.append(guess)

        if verbose:
            logger.info(f"[Turn {turn}] Q: {question}")
            logger.info(f"           A: {answer}")
            logger.info(f"           Guess: {guess}\n")

        if keyword_guessed(guess):
            guessed = True
            score = 20 - turn + 1
            logger.info(f"✅ 정답! 키워드: {ref.keyword} (턴 {turn}, 점수 {score})")
            break

    if not guessed:
        logger.info(f"❌ 20턴 내 실패. 정답은: {ref.keyword}")

    return {"questions": questions, "answers": answers, "guesses": guesses,
            "guessed": guessed, "score": score}


if __name__ == "__main__":
    if ARGS.seed is not None:
        torch.manual_seed(ARGS.seed)

    apply_prompt_variant(ARGS.variant)
    ref.MAX_TURNS = ARGS.max_turns  # guesser_agent의 '마지막 턴' 판정과 max_turns를 일치시킴

    logger.info(f"로그 파일: {LOG_PATH}")
    logger.info(f"프롬프트 variant: {ARGS.variant}")
    logger.info(f"seed: {ARGS.seed}")
    logger.info(f"카테고리: {ref.category} / 정답(디버그): {ref.keyword}\n")

    result = run_game(max_turns=ARGS.max_turns)

    end_time = datetime.now()
    logger.info("\n=== 실행 요약 ===")
    logger.info(f"variant: {ARGS.variant}")
    logger.info(f"시작: {START_TIME:%Y-%m-%d %H:%M:%S}")
    logger.info(f"종료: {end_time:%Y-%m-%d %H:%M:%S}")
    logger.info(f"소요 시간: {end_time - START_TIME}")
    logger.info(f"정답 여부: {result['guessed']} / 점수: {result['score']}")