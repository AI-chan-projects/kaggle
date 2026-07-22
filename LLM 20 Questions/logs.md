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