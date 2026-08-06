"""
llm_20_questions 로컬 실행용 main.py
- 레퍼런스 파일(llm_20_questions.py)의 agents/keyword_guessed 등을 그대로 재사용
- call_llm만 T5 -> Gemma-1.1-2b-it (4bit 양자화)로 교체
- kaggle_environments의 4-agent interpreter는 제출용이라 로컬 테스트엔 과함 ->
  guesser/answerer 한 쌍으로 직접 도는 단순 루프로 대체
"""
import warnings
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 레퍼런스 파일이 llm_20_questions.py 로 저장되어 있다고 가정
# (파일명이 다르면 이 import만 바꿔주면 됨)
import llm_20_questions as ref
from llm_20_questions import GUESSER, ANSWERER, ASK, GUESS, keyword_guessed

# FutureWarning 무시 설정
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="bitsandbytes"
)

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
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
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
            print(f"[Turn {turn}] Q: {question}")
            print(f"           A: {answer}")
            print(f"           Guess: {guess}\n")

        if keyword_guessed(guess):
            guessed = True
            score = 20 - turn + 1
            print(f"✅ 정답! 키워드: {ref.keyword} (턴 {turn}, 점수 {score})")
            break

    if not guessed:
        print(f"❌ 20턴 내 실패. 정답은: {ref.keyword}")

    return {"questions": questions, "answers": answers, "guesses": guesses,
            "guessed": guessed, "score": score}


if __name__ == "__main__":
    print(f"카테고리: {ref.category} / 정답(디버그): {ref.keyword}\n")
    run_game(max_turns=20)