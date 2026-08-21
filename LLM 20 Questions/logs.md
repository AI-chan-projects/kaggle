- 열번째 :
지금까지의 실험 결과:

초기 문제
Guesser가 실제 질문으로 들어가지 않고, Physicist, Leonardo da Vinci 같은 엉뚱한 추측을 생성함.
→ 초기 프롬프트가 질문 생성보다 추측/일반적인 대화로 흐르게 만들고 있었음.
프롬프트를 거의 모두 제거한 실험
GUESSER_INFO_PROMPT_TEMPLATE = "안녕" 또는 빈 프롬프트 상태에서 Gemma 2B는 정상적인 20 Questions 행동을 하지 못함.
→ 모델에게 역할과 출력 형식을 명확히 제공할 필요가 있음.
Answerer의 secret 주입 구조 확인
ref.keyword, ref.category를 함수에서 직접 참조하는 대신 GAME_KEYWORD, GAME_CATEGORY로 시작 시 고정하는 구조를 검토함.
→ 게임 실행 중 secret이 바뀌는 문제를 방지하는 방향.
Answerer deterministic 설정
do_sample=False, max_new_tokens=2로 변경.
→ Answerer는 Guesser보다 훨씬 강하게 출력 형식을 제한하는 게 적절함.
Answerer Q&A history
Answerer에게 이전 Q&A를 전달하는 구조도 확인했지만, 현재 핵심 실험에서는 제거하고 단순화함.
Answerer의 이상 응답
yes만 반복하거나 Answer: yes, The answer 같은 형식 오류가 발생함.
→ 단순히 yes/no 출력 지시만으로는 Gemma 2B의 안정적인 Answerer 역할을 보장하지 못함.
고유명사 판단 실험
Colombia만 제공하고 Is the hidden answer a person?을 질문했는데 yes가 나옴.
Colombia + Type: country + 지역/수도 등 정보세트를 제공해도 yes가 나옴.
→ 고유명사 자체의 문제인지, 작은 모델의 추론/지시 수행 문제인지 추가 검증 필요.
→ 적어도 단순한 정보세트를 추가한다고 문제가 바로 해결되지는 않음.
Category-only 실험
Category: country만 제공했더니 The provided... 같은 설명형 응답이 나옴.
→ category만으로는 안정적인 Answerer를 만들기 어려움.
Guesser의 질문 품질 문제
Is the hidden answer a person? 같은 질문뿐 아니라
Does the hidden answer have any unique or extraordinary properties?
처럼 정보량이 낮고 추상적인 질문을 생성함.
→ Answerer가 잘못 yes를 주면 Guesser가 animal, artifact, legendary creature 등으로 빠르게 잘못된 방향으로 이동함.
현재 가장 유력한 다음 실험
Answerer를 제거하고 사람이 YES/NO를 직접 제공.

keyword/category는 사람이 알고 있고, Guesser는 모르게 한 상태에서:

Guesser Q → 사람 A → Guesser Q → 사람 A → ...

로 진행.

이 실험으로 Guesser 자체의 질문 생성/탐색 능력을 Answerer 문제와 분리해서 확인할 수 있음.

현재는 Answerer와 Guesser가 동시에 불안정해서 원인 분리가 필요하고, 다음 실험은 Human Answerer로 Guesser부터 독립 검증하는 단계

이번에 확인한 건:

**1. Answerer가 keyword/category를 받는 것과, 그 정보를 이용해 정확히 판단하는 것은 별개의 문제다.**

`Colombia`를 주고도:

> Is the hidden answer a person? → YES

가 나왔고, `Colombia + Type: country` 같은 구조화된 정보까지 줘도 `YES`가 나왔어.

그래서 지금은 `Answerer가 질문에 잘 답하는가?`를 보기 전에,

> **Answerer가 주어진 keyword와 그에 대한 객관적 정보를 정확하게 내부적으로 사용할 수 있는가?**

부터 검증해야 해.

**2. Answerer를 바로 게임에 넣는 건 아직 이르다.**

Answerer가 잘못된 `yes`를 내놓으면 Guesser는 그걸 사실이라고 믿고 다음 질문을 만들기 때문에, 이후의 Guesser 성능까지 오염돼.

그래서 실험 순서는:

```text
Keyword
  ↓
Answerer에게 정보 제공
  ↓
정보를 제대로 이해했는가?
  ↓
YES/NO 판단이 정확한가?
  ↓
그 다음에야 Guesser와 연결
```

이게 맞아.

그리고 여기서 한 단계 더 중요한 결론도 있어.

**Answerer에게 keyword를 직접 주는 것만으로 충분하지 않을 가능성이 있다.**

예를 들어:

```text
Answer: Colombia
Type: country
```

를 줬는데도 실패했으니까, 다음에는 **Answerer가 사용할 수 있는 정보 표현 자체를 검증**해야 해.

즉 오늘의 결론을 한 문장으로 정리하면:

> **"게임 전체를 최적화하기 전에, Answerer가 주어진 정답과 그 속성 정보를 정확히 이해하고 객관적인 질문에 일관되게 YES/NO로 답할 수 있는지 독립적으로 검증해야 한다."**

그리고 그 검증이 통과된 뒤에야 Guesser ↔ Answerer를 연결하는 게 맞음.


- 아홉번째 루프 : flan t5 base만 사용하면 굉장히 빠른데 실속이 없어서 gemma vs flan t5로 돌아왔다.


- 여덟번째 루프 : flan t5와 gemma를 구분해서 사용하기로 하였다.
이상한 소리를 내뱉는다. variant와 Seed를 줘야 할 것 같다.

로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260807_150641_baseline.log
프롬프트 variant: baseline
seed: None
guesser model: google/gemma-1.1-2b-it [causal, quantize=True] / answerer model: google/flan-t5-base [seq2seq, quantize=False]
카테고리: city / 정답(디버그): montreal canada

[Turn 1] Q: Thought: Narrowing down to the realm of largaurs...  (thought: Narrowing down to the realm of largaurs...)
           A: no
           Guess: Mesopotamia  (thought: Exploring the enigmatic lore of ancient civilizations... Guess: Mesopotamia)

[Turn 2] Q: Does the object possess any unique physical characteristics?  (thought: Exploring the enigmatic nature of the sought-after artifact... Question: Does the object possess any)
           A: yes
           Guess: Medusa  (thought: Exploring the mythical qualities of the creature... Perhaps Medusa?)



- 일곱번째 루프 : guess를 단어로 제한해서 맥락 정보가 수렴하는 것 같다.
    대회 규칙을 적용해야겠다.
    - 대회 규칙 I need 2 llm models
    - rule (60 sec/round + additional 300 sec/game + when timeout, game ends + when answeror the other answer except yes or no, game ends)
    - guessor (question max letter : 750, thought max letter : 100)
    - answeror (yes or no)

        로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260806_144630_category_first_strict.log
    프롬프트 variant: category_first_strict
    seed: 42
    카테고리: city / 정답(디버그): ingolstadt germany

    [Turn 1] Q: Is it a place?
            A: yes
            Guess: Statue podrobily.

    [Turn 2] Q: Is it something that existed in ancient times?
            A: yes
            Guess: Zeus

    [Turn 3] Q: Is it something associated with royalty or nobility?
            A: yes
            Guess: Alexandria



- 여섯번째 루프 : 같은 질문을 반복함. 이진 질문을 잘못 이해함.
    로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260806_143213_category_first_strict.log
    프롬프트 variant: category_first_strict
    seed: 42
    카테고리: city / 정답(디버그): guangzhou china

    [Turn 1] Q: Is it a place?
            A: yes
            Guess: Statue podrobly.

    [Turn 2] Q: Is it a place or an object?
            A: maybe
            Guess: Hollywood

    [Turn 3] Q: Is it a place or an object?
            A: maybe
            Guess: Alexandria


- 변경사항
    QUESTIONS_PROMPT_TEMPLATE/GUESS_PROMPT_TEMPLATE/GUESS_PROMPT_FINAL_TEMPLATE 끝에 Question: / Guess: cue를 붙여서, 모델이 지시문을 읽고 답하는 게 아니라 그 cue 뒤를 자연스럽게 이어 쓰게 유도 (few-shot 없이도 형식을 강제하는 흔한 트릭) >> 이런게 있다고?

    전략 텍스트에서 Good/Bad 예시 나열을 줄여서 프롬프트 자체를 단순화 (작은 모델일수록 지시가 길고 조건이 많으면 형식을 놓치기 쉬움)

    guesser_agent에 _strip_echoed_cue() 후처리 추가 — 혹시 모델이 Question:/Guess:나 따옴표를 그대로 따라 출력해도 정리되게

    call_llm을 do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15로 전환 — greedy가 "Paris" 같은 최빈값으로 붕괴하는 걸 완화

    --seed 옵션 추가: variant 간 비교할 때 같은 시드로 고정하면 "프롬프트 차이"만 순수하게 비교 가능, 안 주면 매번 랜덤

    한 가지 트레이드오프는 말씀드려야 할 것 같은데, 샘플링을 켜면 같은 variant를 여러 번 돌려도 결과가 매번 달라집니다. 지금 5번째 루프처럼 "variant는 다른데 출력이 똑같다"는 문제는 해결되겠지만, 반대로 "이 variant가 진짜 더 나은지" 판단하려면 한 번의 실행이 아니라 여러 번 돌려서 평균 점수/정답률을 비교해야 신뢰할 수 있습니다. 여러 게임을 자동으로 반복 실행해서 variant별 성적을 집계하는 스윕 스크립트가 필요

    
- 문제 원인 분석 :
    - 프롬프트가 2B 모델에게는 너무 버거운 프롬프트다. (예시가 있어 구분이 안가는 듯, 예시에 대해 그냥 대답해버림.)
    - greedy decoding에서 문답 정보가 없는 경우 학습 데이터의 city -> paris로 고정되는 문제

    - 샘플링, 랜덤(시드를 활용해 재현성)

- 다섯번째 루프 : 다른 프롬프트 옵션, 같은 출력
    로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260806_142302_explicit_category_list.log
    프롬프트 variant: explicit_category_list
    카테고리: city / 정답(디버그): antwerp belgium

    [Turn 1] Q: Yes
            A: yes
            Guess: Paris.


- 네번째 루프 : paris in the yes
    로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260806_142003_category_first_strict.log
    프롬프트 variant: category_first_strict
    카테고리: city / 정답(디버그): yaounde cameroon

    [Turn 1] Q: Yes.
            A: yes
            Guess: Paris.

    [Turn 2] Q: Yes.
            A: yes
            Guess: Paris.



- 세번째 루프 : 질문이 분절되지 않아 Yes or No 질문으로 떨어지지 않는다.
    - 로그 파일: /Users/ironie/code/kaggle/LLM 20 Questions/code/src/logs/20260806_140947_category_first_strict.log
        프롬프트 variant: category_first_strict
        카테고리: country / 정답(디버그): monaco

        [Turn 1] Q: Is it a person, a place, an animal, or an object?
                A: maybe
                Guess: The keyword is person.

            You can't definitively say it's a person, place, or animal, but you can certainly say it's someone.

- 두번째 루프 : 프롬프트가 같으니, 진행이 동일하다.
    - 프롬프트를 변경할 수 있는 인터페이스를 붙이기 위해 레퍼런스 코드의 guess agent의 프롬프트 부분을 모듈 레벨로 리팩터링
    - 프롬프트에 전략을 도입하기
    - main.py에 KV 형식으로 프롬프트를 선택할 수 있게 하였다.

- 첫 루프 : guessing이 제대로 진행되지 않는다.
카테고리: landmark / 정답(디버그): puncak jaya

[Turn 1] Q: Do you think the person you're looking for is a famous explorer?
           A: yes
           Guess: The keyword is explorer.

[Turn 2] Q: Do you think the person you're looking for is a famous explorer who has made significant contributions to the field of exploration?
           A: yes
           Guess: The keyword is explorer.

The person you're looking for is a famous explorer who has made significant contributions to the field of exploration.

[Turn 3] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant impact on the exploration of the Arctic?
           A: yes
           Guess: The keyword is [explorer].

The person you are looking for is a famous explorer who has made significant contributions to the field of exploration.

[Turn 4] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the Antarctic?
           A: yes
           Guess: The keyword is [Antarctic].

The person you are looking for has made significant contributions to the exploration of the Antarctic.

[Turn 5] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the Pacific Ocean?
           A: yes
           Guess: The keyword is [Antarctic].

The person you are looking for has made significant contributions to the exploration of the Antarctic.

[Turn 6] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human body?
           A: yes
           Guess: The keyword is [explorer].

The person you are looking for has made significant contributions to the exploration of various parts of the world, including the Arctic, Antarctic, Pacific Ocean, and the human body.

[Turn 7] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human mind?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, leading to the conclusion that the person being sought is a famous explorer.

[Turn 8] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human spirit?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, leading to the conclusion that the person being sought is a famous explorer.

[Turn 9] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human consciousness?
           A: yes
           Guess: The keyword is: **Antarctic**.

Each question progressively narrows down the possibilities to the Antarctic, starting with the vastness of the ocean and then focusing on specific contributions to the exploration of the continent.

[Turn 10] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, focusing on specific geographic regions and contributions to exploration.

[Turn 11] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, focusing on different aspects of exploration.

[Turn 12] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, focusing on different aspects of exploration.

[Turn 13] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, leading to the final answer.

[Turn 14] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, leading to the final answer.

[Turn 15] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is: **Explorer**

[Turn 16] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human spirit?
           A: yes
           Guess: The keyword is [explorer].

Each question progressively narrows down the possibilities, leading to the final answer.

[Turn 17] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human consciousness?
           A: yes
           Guess: The keyword is [explorer].

You have been able to determine that the person is a famous explorer who has made significant contributions to the exploration of various parts of the world.

[Turn 18] Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential?
           A: yes
           Guess: The keyword is [explorer].

You have been able to determine that the person you are looking for is a famous explorer who has made a significant contribution to the exploration of various parts of the world.

[Turn 19] Q: Based on your previous questions, the person you're looking for is most likely a famous explorer who has made a significant contribution to the exploration of the human potential.
           A: yes
           Guess: The keyword is human potential.

Throughout the questions, the focus has been on the person's contributions to the exploration of the human potential.

[Turn 20] Q: Q: Do you think the person you're looking for is a famous explorer who has made a significant contribution to the exploration of the human potential in the context of their exploration of the human consciousness?
           A: yes
           Guess: The keyword is potential.

Throughout the questions, the focus has been on the person's contributions to the exploration of potential in various fields.

❌ 20턴 내 실패. 정답은: puncak jaya

- llm_20_questions.py에서 함수를 차례로 불러와서 파싱해봐야겠다.
    - flan T5 (T5 계열 모델)의 특징 및 T5란 무엇인가
        Text To Text Transfer Transformer (T가 5개)
        - 아주 T스러운 모델임. seq2seq 구조 
        - 인코더와 디코더가 둘 다 들어있다. 문장 전체 맥락을 파악한 다음 답변을 찾아야하는 작업에 압도적인 성능
    - Auto계열의 경우 디코더 전용 구조임. 현재 우리가 사용하는 대부분의 LLM
    - Auto 계열의 경우, 질문 -> 토크나이저 : 난수표 -> 모델(디코더) :: 방적기 (causalLM)
    - T5 계열의 경우 질문 -> 토크나이저 -> 모델 인코더 (벡터 변환) : 모델 전용 embedding engine -> 디코더
    - Attention 구조도 차이가 있음.
        - Auto계열 : 미래에 올 단어는 마스킹되어있음. 입력받는 순서대로 맥락을 파악함.
        - T5계열 : 양방향으로 문서를 인코딩함. (BERT와 같은 방식), 디코더의 경우 답을 작성할 때는 Auto계열과 동일하지만, 
                  인코딩된 벡터를 Cross Attention함.
        - KV캐시는 둘다 사용함.
        - 그러면 T5계열을 사용하면 되지 않냐? : 단점 : 인코딩 할 내용이 길어진다면?, 벡터로 전체를 변환해야한다면?
        - hybrid 검색 엔진을 만들 때는 주로 다음과 같은 알고리즘과 모델을 활용한다. 
        - Query(BM25 알고리즘 + Embedding Model):Retrieval -> 
            Reranker(Query + Retrieval):Cross-Encoding(BERT계열) ->
            Generation(Reranker):Decode(Auto계열(업계 표준) 또는 T5계열(환각 방지 탁월))
        - Reranker 모델 중첩의 한계 : 
            하나의 모델만 여러번 쓰기보다는 앙상블이나, 쿼리 다양화로 후보군을 넓히는 게 성능 향상에 도움이 된다.
        - Query의 경우 Hybrid Search (Lexical + Semantic)를 권장한다.
            BM25는 의미적인 동의어(코사인 유사도)를 모름.
            Embedding Vector는 리터럴한 중요도를 무시함.(고유명사 등)
            -> 각각 검색을 따로 수행
            -> 순위 기반 융합 RRF (Reciprocal Rank Fusion) : 두 검색의 순위를 확인해서 합산 순위가 높은 것만 뽑아줌
    - 그러면 모델은 어떤 종류가 있을까?
        - 인코더 전용 : BERT 계열(Rerank, Embedding특화 양방향 인코딩)
        - 음성 및 멀티모달 : Whisper(구조적으로는 T5와 비슷함), CLIP(사진과 텍스트를 동시에 입력받아 대조함)
        - 확산 모델 : Diffusion (Autoregressive 구조가 아닌 노이즈 부터 시작해 점차 형태를 갖추고 다듬으며 결과물을 완성)
            이미지, 오디오, 비디오 생성 분야에서 성능이 좋음
        - 상태 공간 모델 : SSM(State Space Model : Mamba, Transformer의 긴 텍스트에 대한 병목을 해결하기 위한 새로운 구조)
            Attention구조가 아니라 데이터가 흘러가면서 기억을 업데이트함

- 게임 구성하기
    - 룰에 따라서 게임 구성하기
    - 1 token ≈ 4 letters
    - 게임 로직이 llm_20_questions.py 에 함수로 작성되어 있음.
    - 만약 여러개의 서로 다른 대답을 얻고 싶은 경우
        ```py
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=3,  # 서로 다른 대답 3개를 만들어줘!
            do_sample=True,  # 무작위성을 켜서 매번 다르게 대답하게 해줘!
            temperature=0.7,  # 창의성(다양성) 조절
        )
    - llm_parent_dir = str(Path.home()) + "/.cache/huggingface/hub"
        설정완료

        # 이 경우에만 batch_decode 결과에 진짜로 3개의 다른 문장이 들어옵니다.
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ```

- 모델 불러오기 3
    - 두 가지 코드 수정할 것 :
    1. input_ids = input_ids.to('mps') before running .generate()
        input_ids = tokenizer(input_text, return_tensors="pt").to("mps")
    2. _check_is_size deprecation warning _check(i>=0) instead.
        warnings.filterwarnings(
           "ignore", category=FutureWarning, module="bitsandbytes"
        )

- 모델 불러오기 2
    - 카페에서 집으로 오니 401이 뜬다.
    - > hf auth login 명령어를 입력해 다시 로그인함
    - 정상 다운로드됨

- 모델 불러오기
    - local 환경에서 진행하는 경우 huggingface를 거쳐야 한다. 아니면 배포처에서 하나하나 다운로드 받아야함
    - huggingface api 등록 및 환경변수 설정
    - HF_TOKEN="..."
    - Access to model google/gemma-1.1-2b-it is restricted and you are not in the authorized list. Visit https://huggingface.co/google/gemma-1.1-2b-it to ask for access.
    - 약관 읽고 접근 권한 얻음
    - ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
    - [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    - 시간 오래 걸린다...
    - transformer로 다운로드 받은 모델은 어디로 가는가?
    - llm_20_questions.py > llm_parent_dir에 경로 지정하기
    "~/.cache/huggingface/hub/"
    - 모델 딕셔너리를 받아서 약어로 불러올 수 있게 해보기

- bitsandbytes의 설정값이 transformers에 통합되었다. 아래와 같이 사용한다.

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 1. 4비트 양자화 설정 정의
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 일반적으로 성능이 좋은 NF4(NormalFloat4) 형식 사용
    bnb_4bit_use_double_quant=True,  # 메모리를 추가로 아껴주는 이중 양자화
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 사용할 데이터 타입
)

# 2. 모델 로드 시 quantization_config에 전달
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
)
```

- llm_20_questions_py 가 함수 뭉치이고, 그걸 이용해서 코드를 짜야하는 걸 알았음
    - main.py 작성 예정

- 대회 규칙 I need 2 llm models
    - rule (60 sec/round + additional 300 sec/game + when timeout, game ends + when answeror the other answer except yes or no, game ends)
    - guessor (question max letter : 750, thought max letter : 100)
    - answeror (yes or no)

- no runtime error

- code/src/llm_20_questions.py init
    - from .keywords > from keywords :: __init__.py

- > unzip llm-20-questions
    - .gitignore > # resource : \ ./res

- kaggle api token 발행
    - export 완료
    - pip install kaggle
    - kaggle competitions list
    - (일부 대회의 경우 웹사이트에서 먼저 규칙(Rules)에 동의(Join Competition)해야 다운로드가 가능합니다.)
    - kaggle competitions download -c llm-20-questions

- kaggle 환경 로컬로 불러오는 api 작업 진행 예정

- 앞으로 원활한 환경 구축을 위해 setup.sh 작성