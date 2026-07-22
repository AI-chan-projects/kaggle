- 게임 구성하기
    - 

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