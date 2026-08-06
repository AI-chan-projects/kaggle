import json
import os
import pandas as pd
import random
import string
import torch

from keywords import KEYWORDS_JSON
from os import path
from pathlib import Path
from random import choice
from string import Template
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Hugging Face llm dir : ~/.cache/huggingface/hub
llm_parent_dir = str(Path.home()) + "/.cache/huggingface/hub"
# llm_parent_dir = "/kaggle/input/flan-t5/pytorch/large"

device = None
model = None
tokenizer = None
model_initialized = False

ERROR = "ERROR"
DONE = "DONE"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"
TIMEOUT = "TIMEOUT"

GUESS = "guess"
ASK = "ask"
GUESSER = "guesser"
ANSWERER = "guesser"

MAX_TURNS = 20  # main.py의 run_game(max_turns=...)과 맞춰서 사용
EARLY_TURN_THRESHOLD = 3

# ---- 프롬프트 템플릿(전역 상수) ----
# call_llm과 같은 방식으로 몽키패치 가능: main.py에서
# `ref.QUESTION_STRATEGY_EARLY = "..."` 처럼 값만 덮어쓰면
# guesser_agent/answerer_agent 코드는 그대로 두고 프롬프트만 바꿔서 테스트 가능.

GUESSER_INFO_PROMPT_TEMPLATE = """You are playing a game of 20 questions where you ask the questions and try to figure out the keyword, which will be a real or fictional person, place, or thing. \nHere is what you know so far:\n{q_a_thread}"""

QUESTION_STRATEGY_EARLY = (
    "Strategy: this is still an early question. Test exactly ONE hypothesis "
    "at a time, and phrase it so it is answerable with a strict yes or no. "
    "NEVER list multiple options joined by 'or' in a single question — that "
    "cannot be answered with yes/no. "
    "Good example: 'Is it a place?' "
    "Bad example: 'Is it a person, a place, an animal, or an object?' "
    "Pick the single most likely broad category (person, place, animal, or "
    "object) and ask about that one category alone."
)

QUESTION_STRATEGY_LATE = (
    "Strategy: use everything you know so far to narrow down further within "
    "the category/subcategory you've already identified. Test exactly ONE "
    "hypothesis at a time, phrased so it is answerable with a strict yes or "
    "no — never list multiple options joined by 'or' in a single question. "
    "Avoid repeating a question that is logically the same as one already "
    "asked."
)

QUESTIONS_PROMPT_TEMPLATE = """Ask one yes or no question. This is question {turn_num} of {max_turns}.
Your question must be answerable with only "yes" or "no" — do not phrase it as a multiple-choice question.
{strategy}"""

GUESS_PROMPT_TEMPLATE = """Guess the keyword. Only respond with the exact word/phrase. For example, if you think the keyword is [paris], don't respond with [I think the keyword is paris] or [Is the kewyord Paris?]. Respond only with the word [paris]."""

GUESS_PROMPT_FINAL_TEMPLATE = (
    "This is your LAST guess. Based on everything above, give your single "
    "best guess for the keyword even if you are not fully certain. "
    "Only respond with the exact word/phrase. For example, if you think the "
    "keyword is [paris], don't respond with [I think the keyword is paris] "
    "or [Is the keyword Paris?]. Respond only with the word [paris]."
)

ANSWERER_INFO_PROMPT_TEMPLATE = """You are a very precise answerer in a game of 20 questions. The keyword that the questioner is trying to guess is [the {category} {keyword}]. """

ANSWER_QUESTION_PROMPT_TEMPLATE = """Answer the following question with only yes, no, or if unsure maybe: {question}"""

keywords_list = json.loads(KEYWORDS_JSON)
keyword_cat = random.choice(keywords_list)
category = keyword_cat["category"]
keyword_obj = random.choice(keyword_cat["words"])
keyword = keyword_obj["keyword"]
alts = keyword_obj["alts"]


def guesser_agent(obs):
    q_a_thread = ""
    for i in range(0, len(obs.answers)):
        q_a_thread = "{}Q: {} A: {}\n".format(
            q_a_thread,
            obs.questions[i],
            obs.answers[i]
        )

    if obs.turnType == ASK:
        # obs.questions는 아직 이번 질문을 담기 전이므로, 다음 질문 번호 = len(questions) + 1
        turn_num = len(obs.questions) + 1
        strategy = QUESTION_STRATEGY_EARLY if turn_num <= EARLY_TURN_THRESHOLD else QUESTION_STRATEGY_LATE

        questions_prompt = QUESTIONS_PROMPT_TEMPLATE.format(
            turn_num=turn_num, max_turns=MAX_TURNS, strategy=strategy
        )

        prompt = "{}{}".format(
            GUESSER_INFO_PROMPT_TEMPLATE.format(q_a_thread=q_a_thread),
            questions_prompt
        )
    elif obs.turnType == GUESS:
        # obs.questions는 이번 라운드 질문까지 이미 포함된 상태이므로, 현재 라운드 = len(questions)
        turn_num = len(obs.questions)
        guess_prompt = GUESS_PROMPT_FINAL_TEMPLATE if turn_num >= MAX_TURNS else GUESS_PROMPT_TEMPLATE

        prompt = "{}{}".format(
            GUESSER_INFO_PROMPT_TEMPLATE.format(q_a_thread=q_a_thread),
            guess_prompt
        )
    else:
        return ""

    return call_llm(prompt)



def answerer_agent(obs):
    if obs.turnType == "answer":
        prompt = "{}{}".format(
            ANSWERER_INFO_PROMPT_TEMPLATE.format(category=category, keyword=keyword),
            ANSWER_QUESTION_PROMPT_TEMPLATE.format(question=obs.questions[-1])
        )
        return call_llm(prompt)
    else: 
        return ""


agents = {GUESSER: guesser_agent, ANSWERER: answerer_agent}

def guesser_action(active, inactive, step):
    guessed = False
    if not active.action:
        active.status = ERROR
    elif active.observation.turnType == ASK:
        question = active.action[:2000]
        active.observation.questions.append(question)
        inactive.observation.questions.append(question)
    elif active.observation.turnType == GUESS:
        guess = active.action[:100]
        active.observation.guesses.append(guess)
        inactive.observation.guesses.append(guess)
    if active.action and keyword_guessed(active.action):
        guessed = True
        score = 20 - int(step / 3)
        end_game(active, inactive, score, DONE, DONE)
    return guessed

def end_game(active, inactive, reward, status, inactive_status):
    active.observation.keyword = keyword
    active.observation.category = category
    inactive.observation.keyword = keyword
    inactive.observation.category = category
    active.reward = reward
    inactive.reward = reward
    active.status = status
    inactive.status = inactive_status


def answerer_action(active, inactive):
    active.observation.keyword = keyword
    active.observation.category = category
    response = active.action
    if not response:
        response = "none"
        end_game(active, inactive, -1, ERROR, DONE)
    elif "yes" in response.lower():
        response = "yes"
    elif "no" in response.lower():
        response = "no"
    else:
        response = "maybe"
        end_game(active, inactive, -1, ERROR, DONE)
    active.observation.answers.append(response)
    inactive.observation.answers.append(response)

def increment_turn(active, inactive, step, guessed):
    if step == 59 and not guessed:
        end_game(active, inactive, -1, DONE, DONE)
    elif active.observation.turnType == "guess":
        active.observation.turnType = "ask"
    elif active.observation.turnType == "ask":
        active.observation.turnType = "guess"
        active.status = INACTIVE
        inactive.status = ACTIVE
    else:
        active.status = INACTIVE
        inactive.status = ACTIVE


def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active1 = state[0] if state[0].status == ACTIVE else state[1]
    inactive1 = state[0] if state[0].status == INACTIVE else state[1]
    active2 = state[2] if state[2].status == ACTIVE else state[3]
    inactive2 = state[2] if state[2].status == INACTIVE else state[3]
    if active1.status == DONE and inactive1.status == DONE:
        active1 = None
        inactive1 = None
    if active2.status == DONE or inactive2.status == DONE:
        active2 = None
        inactive2 = None
    if active1 is None and inactive1 is None and active2 is None and inactive2 is None:
        return state

    step = state[0].observation.step

    end_early = (active1 and active1.status) in (TIMEOUT, ERROR) or (active2 and active2.status in (TIMEOUT, ERROR))
    either_guessed = False

    if active1 is not None:
        guessed = False
        if active1.observation.role == GUESSER:
            guessed = guesser_action(active1, inactive1, step)
            either_guessed = guessed
        else:
            answerer_action(active1, inactive1)
        if active1.status in (TIMEOUT, ERROR):
            end_game(active1, inactive1, 0, active1.status, DONE)
        elif end_early:
            end_game(active1, inactive1, 0, DONE, DONE)
        else:
            increment_turn(active1, inactive1, step, guessed)

    if active2 is not None:
        guessed = False
        if active2.observation.role == GUESSER:
            guessed = guesser_action(active2, inactive2, step)
            either_guessed = either_guessed or guessed
        else:
            answerer_action(active2, inactive2)
        if active2.status in (TIMEOUT, ERROR):
            end_game(active2, inactive2, 0, active2.status, DONE)
        elif end_early:
            end_game(active2, inactive2, 0, DONE, DONE)
        else:
            increment_turn(active2, inactive2, step, guessed)

    return state


def renderer(state, env):

    for s in state:
        print("role: ", s.observation.role)
        if s.observation.role == GUESSER:
            transcript = ""
            for i in range(0, len(s.observation.guesses)):
                transcript = "{}Q: {} A: {}\nG: {}\n".format(
                    transcript, s.observation.questions[i],
                    s.observation.answers[i],
                    s.observation.guesses[i]
                )
            print(transcript)

        print("keyword: ", s.observation.keyword)
        print("score: ", s.reward)
        print("")
        print("")
        print("")

    return ""


jsonpath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.json"))
with open(jsonpath) as f:
    specification = json.load(f)

def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.js"))
    with open(jspath) as f:
        return f.read()


def keyword_guessed(guess: str) -> bool:
    def normalize(s: str) -> str:
      t = str.maketrans("", "", string.punctuation)
      return s.lower().replace("the", "").replace(" ", "").translate(t)

    if normalize(guess) == normalize(keyword):
      return True
    for s in alts:
      if normalize(s) == normalize(guess):
        return True

    return False


def call_llm(prompt: str) -> str:
    global model_initialized
    global device
    global model
    global tokenizer

    if not model_initialized:
        if os.path.exists(llm_parent_dir) and len(os.listdir(llm_parent_dir)) > 0:
            dirs = os.listdir(llm_parent_dir)
            llm_dir = "{}/{}".format(llm_parent_dir, dirs[0])
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = T5ForConditionalGeneration.from_pretrained(llm_dir).to(device)
            tokenizer = T5Tokenizer.from_pretrained(llm_dir)
            model_initialized = True
        else:
            print("t5-flan model required to use default agents. Add any version of the large model.")
            print("https://www.kaggle.com/models/google/flan-t5/frameworks/pyTorch/variations/large.")
            raise Exception("t5-flan model required to use default agents.")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]