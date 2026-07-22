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