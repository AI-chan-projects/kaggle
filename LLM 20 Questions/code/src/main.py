"""
llm_20_questions 로컬 실행용 main.py (Kaggle 공식 환경 턴 규칙 반영)

설계 원칙: 레퍼런스 파일(llm_20_questions.py)은 원본 그대로 두고 건드리지 않는다.
실제 Kaggle 환경의 턴 타입(ask, answer, guess) 흐름에 맞춰 로컬 게임 루프를 동기화함.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"bitsandbytes\..*",
)

import argparse
import logging
import re
import string
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

import llm_20_questions as ref  # category / keyword / alts / keyword_guessed()만 사용

# ============================================================
# 게임 규칙 상수
# ============================================================
DEFAULT_MAX_TURNS = 20
ROUND_TIMEOUT_SEC = 10000       # 질문/답변/추측 1회 호출당 제한 시간
GAME_TIMEOUT_SEC = 100000      # 게임 전체 누적 제한 시간
QUESTION_MAX_CHARS = 7500     # guesser의 question/guess 최대 길이
THOUGHT_MAX_CHARS = 1000      # guesser의 thought 최대 길이

GAME_KEYWORD = ref.keyword
GAME_CATEGORY = ref.category

class RoundTimeoutError(Exception):
    """한 번의 LLM 호출이 ROUND_TIMEOUT_SEC를 넘겼을 때."""


# ============================================================
# 프롬프트 템플릿
# ============================================================
GUESS_PROMPT_TEMPLATE = """
You are the guesser in a game of 20 Questions.

Previous questions and answers:

{q_a_thread}

Based on the previous questions and answers, make your best guess about the hidden answer.

Guess:
"""

GUESSER_INFO_PROMPT_TEMPLATE = """
You are the guesser in a game of 20 Questions.

Previous questions and answers:

{q_a_thread}
"""

QUESTION_STRATEGY_EARLY = (
    "Ask one simple yes/no question that identifies a broad property "
    "of the hidden answer. Do not guess the answer yet."
)

QUESTION_STRATEGY_LATE = (
    "Ask one new yes/no question that narrows down the hidden answer. "
    "Use the previous questions and answers. "
    "Do not repeat a previous question. "
    "Do not ask about personal experiences, opinions, feelings, or preferences. "
    "Ask only about objective properties of the hidden answer."
)

QUESTIONS_PROMPT_TEMPLATE = """
{strategy}

Question:
"""

# ---- seq2seq(T5/Flan-T5 등)용 단순 버전 ----
QUESTIONS_PROMPT_TEMPLATE_SEQ2SEQ = """
Previous questions and answers:

{q_a_thread}

{strategy}

Output one yes/no question:
"""

GUESS_PROMPT_TEMPLATE_SEQ2SEQ = """
Previous questions and answers:

{q_a_thread}

Make your best guess about the hidden answer.

Output only the guess:
"""

ANSWERER_INFO_PROMPT_TEMPLATE = """
Answer the question using the information below.

The hidden answer is:

Colombia

Type: country
Continent: South America
Region: Northwestern South America
Has coastline: yes
Landlocked: no
Capital: Bogotá
"""

ANSWER_QUESTION_PROMPT_TEMPLATE = """
Question:

{question}

Respond with only YES or NO.
"""


# ============================================================
# 프롬프트 실험용 인터페이스
# ============================================================
PROMPT_VARIANTS = {
    "baseline": {},
    "category_first_strict": {
        "QUESTION_STRATEGY_EARLY": (
            ""
        ),
    },
}


def apply_prompt_variant(name: str) -> None:
    if name not in PROMPT_VARIANTS:
        raise ValueError(f"알 수 없는 variant: {name} (선택 가능: {list(PROMPT_VARIANTS)})")
    g = globals()
    for attr, value in PROMPT_VARIANTS[name].items():
        if attr not in g:
            raise AttributeError(f"main.py에 '{attr}' 전역이 없습니다.")
        g[attr] = value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline", choices=list(PROMPT_VARIANTS))
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guesser-model", default="google/gemma-1.1-2b-it")
    parser.add_argument("--guesser-family", default="causal", choices=["causal", "seq2seq"])
    parser.add_argument("--guesser-quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answerer-model", default="google/gemma-1.1-2b-it")
    parser.add_argument("--answerer-family", default="causal", choices=["causal", "seq2seq"])
    parser.add_argument("--answerer-quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    return parser.parse_args()


ARGS = parse_args()

# ============================================================
# 로깅 설정
# ============================================================
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

# ============================================================
# 모델 로딩
# ============================================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)


def load_model_handle(model_name: str, family: str, quantize: bool, device: str) -> SimpleNamespace:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cls = AutoModelForSeq2SeqLM if family == "seq2seq" else AutoModelForCausalLM
    dtype = torch.bfloat16 if quantize else torch.float32
    device_kwargs = {"device_map": "auto"} if device == "auto" else {"device_map": {"": device}}
    model = model_cls.from_pretrained(
        model_name, dtype=dtype,
        quantization_config=quantization_config if quantize else None,
        **device_kwargs,
    )
    return SimpleNamespace(tokenizer=tokenizer, model=model, family=family)


guesser_handle = load_model_handle(ARGS.guesser_model, ARGS.guesser_family, ARGS.guesser_quantize, ARGS.device)

if (
    ARGS.answerer_model == ARGS.guesser_model
    and ARGS.answerer_family == ARGS.guesser_family
    and ARGS.answerer_quantize == ARGS.guesser_quantize
):
    answerer_handle = guesser_handle
else:
    answerer_handle = load_model_handle(ARGS.answerer_model, ARGS.answerer_family, ARGS.answerer_quantize, ARGS.device)


def _generate(handle: SimpleNamespace, prompt: str, gen_kwargs: dict, timeout_sec: int) -> str:
    def _run(result: dict):
        try:
            if handle.family == "seq2seq":
                inputs = handle.tokenizer(prompt, return_tensors="pt").to(handle.model.device)
                output_ids = handle.model.generate(**inputs, **gen_kwargs)
                result["value"] = handle.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            else:
                chat = [{"role": "user", "content": prompt}]
                inputs = handle.tokenizer.apply_chat_template(
                    chat, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(handle.model.device)
                input_len = inputs["input_ids"].shape[-1]
                output_ids = handle.model.generate(**inputs, **gen_kwargs)
                new_tokens = output_ids[0][input_len:]
                result["value"] = handle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            result["error"] = e

    result: dict = {}
    thread = threading.Thread(target=_run, args=(result,), daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise RoundTimeoutError(f"{timeout_sec}초 제한 초과")
    if "error" in result:
        raise result["error"]
    return result["value"]


def _parse_thought_and_response(raw: str, label: str, max_chars: int):
    text = raw.strip()
    label_pos = re.search(rf"{label}\s*:\s*", text, flags=re.IGNORECASE)

    if label_pos:
        before, after = text[:label_pos.start()], text[label_pos.end():]
        response = after.strip().splitlines()[0].strip() if after.strip() else ""
    else:
        before = ""
        response = re.sub(r"^Thought\s*:\s*", "", text, flags=re.IGNORECASE)
        response = response.strip().splitlines()[0].strip() if response.strip() else response.strip()

    thought_match = re.search(r"Thought\s*:\s*(.*)", before, flags=re.IGNORECASE)
    thought = thought_match.group(1).strip()[:THOUGHT_MAX_CHARS] if thought_match else ""
    response = response.strip('"\'* ')[:max_chars]
    return thought, response


def build_q_a_thread(questions, answers):
    thread = ""
    for q, a in zip(questions, answers):
        thread += f"Q: {q} A: {a}\n"
    return thread


def _clean_plain_response(raw: str, max_chars: int) -> str:
    text = raw.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line.strip('"\'* ')[:max_chars]


# ============================================================
# 에이전트 행동 함수 (Kaggle 규격 맞춤형)
# ============================================================
def guesser_ask(questions, answers):
    turn_num = len(questions) + 1
    strategy = QUESTION_STRATEGY_EARLY if turn_num <= 1 else QUESTION_STRATEGY_LATE
    prompt = GUESSER_INFO_PROMPT_TEMPLATE.format(q_a_thread=build_q_a_thread(questions, answers))

    if guesser_handle.family == "seq2seq":
        prompt += QUESTIONS_PROMPT_TEMPLATE_SEQ2SEQ.format(
            turn_num=turn_num, max_turns=ARGS.max_turns, strategy=strategy,
            question_max=QUESTION_MAX_CHARS,
        )
    else:
        prompt += QUESTIONS_PROMPT_TEMPLATE.format(
            q_a_thread=build_q_a_thread(questions, answers),
            turn_num=turn_num, max_turns=ARGS.max_turns, strategy=strategy,
            thought_max=THOUGHT_MAX_CHARS, question_max=QUESTION_MAX_CHARS,
        )

    gen_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15, max_new_tokens=200)
    raw = _generate(guesser_handle, prompt, gen_kwargs, ROUND_TIMEOUT_SEC)

    if guesser_handle.family == "seq2seq":
        return "", _clean_plain_response(raw, QUESTION_MAX_CHARS)
    return _parse_thought_and_response(raw, "Question", QUESTION_MAX_CHARS)


def guesser_guess(questions, answers):
    prompt = GUESSER_INFO_PROMPT_TEMPLATE.format(q_a_thread=build_q_a_thread(questions, answers))

    if guesser_handle.family == "seq2seq":
        prompt += GUESS_PROMPT_TEMPLATE_SEQ2SEQ.format(question_max=QUESTION_MAX_CHARS)
    else:
        prompt += GUESS_PROMPT_TEMPLATE.format(q_a_thread=build_q_a_thread(questions, answers), thought_max=THOUGHT_MAX_CHARS, question_max=QUESTION_MAX_CHARS)

    gen_kwargs = dict(do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.1, max_new_tokens=200)
    raw = _generate(guesser_handle, prompt, gen_kwargs, ROUND_TIMEOUT_SEC)

    if guesser_handle.family == "seq2seq":
        return "", _clean_plain_response(raw, QUESTION_MAX_CHARS)
    return _parse_thought_and_response(raw, "Guess", QUESTION_MAX_CHARS)


def answerer_answer(question, questions, answers) -> str:
    prompt = ANSWERER_INFO_PROMPT_TEMPLATE.format(
        category=GAME_CATEGORY,
        keyword=GAME_KEYWORD,
    )

    prompt += ANSWER_QUESTION_PROMPT_TEMPLATE.format(
        question=question
    )

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=2,
    )

    logger.info(
        f"ANSWERER DEBUG | keyword={GAME_KEYWORD!r} | "
        f"category={GAME_CATEGORY!r}"
    )

    logger.info(f"ANSWERER PROMPT:\n{prompt}")

    return _generate(
        answerer_handle,
        prompt,
        gen_kwargs,
        ROUND_TIMEOUT_SEC,
    ).strip()


# ============================================================
# 게임 루프 (Kaggle 공식 환경 스텝 규칙 반영)
# ============================================================
def run_game(max_turns: int = DEFAULT_MAX_TURNS, verbose: bool = True) -> dict:
    questions, answers, guesses = [], [], []
    guessed = False
    score = 0
    status = "IN_PROGRESS"
    game_start = time.monotonic()

    def elapsed():
        return time.monotonic() - game_start

    # Kaggle 20 Questions 규칙: 총 max_turns 동안 (질문 -> 답변 -> 추측) 순서로 진행됨
    for turn in range(1, max_turns + 1):
        # 1) Guesser가 질문(Ask) 생성
        try:
            thought, question = guesser_ask(questions, answers)
        except RoundTimeoutError:
            status = "TIMEOUT"
            logger.info(f"⏱️ 턴 {turn}: 질문 생성이 제한시간 초과")
            break
        
        if elapsed() > GAME_TIMEOUT_SEC:
            status = "TIMEOUT"
            break
        
        questions.append(question)

        # 2) Answerer가 답변(Answer) 생성 (yes/no 검증)
        try:
            raw_answer = answerer_answer(question, questions, answers)
        except RoundTimeoutError:
            status = "TIMEOUT"
            logger.info(f"⏱️ 턴 {turn}: 답변 생성이 제한시간 초과")
            break

        normalized = raw_answer.strip().lower().strip(string.punctuation + " ")
        if normalized not in ("yes", "no"):
            status = "INVALID_ANSWER"
            logger.info(f"🚫 턴 {turn}: answerer가 yes/no 이외로 응답('{raw_answer}') -> 규칙 위반, 게임 종료")
            break
        
        answers.append(normalized)

        if elapsed() > GAME_TIMEOUT_SEC:
            status = "TIMEOUT"
            break

        # 3) Guesser가 추측(Guess) 시도
        try:
            guess_thought, guess = guesser_guess(questions, answers)
        except RoundTimeoutError:
            status = "TIMEOUT"
            logger.info(f"⏱️ 턴 {turn}: 추측 생성이 제한시간 초과")
            break
        
        guesses.append(guess)

        if verbose:
            logger.info(f"[Turn {turn}] Q: {question} (thought: {thought})")
            logger.info(f"           A: {normalized}")
            logger.info(f"           Guess: {guess} (thought: {guess_thought})\n")

        if elapsed() > GAME_TIMEOUT_SEC:
            status = "TIMEOUT"
            break

        # 정답 판정 (Kaggle 함수 ref.keyword_guessed 이용)
        if ref.keyword_guessed(guess):
            guessed = True
            score = max_turns - turn + 1
            status = "DONE"
            logger.info(f"✅ 정답! 키워드: {ref.keyword} (턴 {turn}, 점수 {score})")
            break

    if status == "IN_PROGRESS":
        status = "DONE"
        logger.info(f"❌ {max_turns}턴 내 실패. 정답은: {ref.keyword}")

    return {
        "questions": questions, "answers": answers, "guesses": guesses,
        "guessed": guessed, "score": score, "status": status,
        "elapsed_sec": elapsed(),
    }


if __name__ == "__main__":
    if ARGS.seed is not None:
        torch.manual_seed(ARGS.seed)

    apply_prompt_variant(ARGS.variant)

    logger.info(f"로그 파일: {LOG_PATH}")
    logger.info(f"프롬프트 variant: {ARGS.variant}")
    logger.info(f"seed: {ARGS.seed}")
    logger.info(
        f"guesser model: {ARGS.guesser_model} [{ARGS.guesser_family}, "
        f"quantize={ARGS.guesser_quantize}] / "
        f"answerer model: {ARGS.answerer_model} [{ARGS.answerer_family}, "
        f"quantize={ARGS.answerer_quantize}]"
    )
    logger.info(f"카테고리: {ref.category} / 정답(디버그): {ref.keyword}\n")

    result = run_game(max_turns=ARGS.max_turns)

    end_time = datetime.now()
    logger.info("\n=== 실행 요약 ===")
    logger.info(f"variant: {ARGS.variant}")
    logger.info(f"상태: {result['status']}")
    logger.info(f"시작: {START_TIME:%Y-%m-%d %H:%M:%S}")
    logger.info(f"종료: {end_time:%Y-%m-%d %H:%M:%S}")
    logger.info(f"소요 시간: {end_time - START_TIME}")
    logger.info(f"정답 여부: {result['guessed']} / 점수: {result['score']}")