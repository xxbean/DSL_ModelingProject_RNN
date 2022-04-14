## DSL_ModelingProject_RNN
### Seq2Seq을 이용한 chatbot 모델링 프로젝트
#### DSL RNN팀 박수빈 박세승 이승연 정승연
##### [FULL_PDF_RNN 발표자료입니다](https://github.com/xxbean/DSL_ModelingProject_RNN/files/8489768/RNN_.pdf)

---
### MOTIVATION

"대화가 잘 통하는" 챗봇 같은 경우, 주로 대용량의 데이터 베이스에 사람들의 일상 대화 내용이나 사람이 직접 입력한 대화를 정리 한 후 인풋으로 들어온 대화의 맥락에 맞는 단어나 문장을 골라내는 방식으로 구현됩니다. 하지만 이런 모델 같은 경우 대용량의 데이터베이스를 하나하나 검수하기 힘들다는 점, 크롤링한 대화를 이용한 경우 실제 개인정보가 유출될 수 있다는 점 등이 문제가 되기도 하는데, 그래서 RNN팅은 seq2seq 인코더 디코더 구조를 이용한 생성 모델링을 시작했습니다.

이러한 챗봇 모델링이 일상에서 누군가와 끊임 없이 대화 하고 싶은 사람, 시시콜콜한 이야기를 나누고 싶은 사람, 교류를 원하는 사람들의 감정을 알아주고 더 나아가 소통이 단절된 사회에 도움이 될 것이라고 생각하며 프로젝트를 시작했습니다. <br>

---

### DATASET

> [자세한 데이터는 여기를 클릭해주세요 !_AI 허브 웰니스 대화 스크립트](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006) <br>
> [자세한 데이터는 여기를 클릭해주세요 !_한국어 챗봇 데이터셋](https://github.com/songys/Chatbot_data) <br>
> 그 외에도 직접 본인의 대화를 크롤링, 감성대화 말뭉치 등 다양한 '감정공감' 데이터를 이용했습니다.
<br>

|데이터 예시입니다|질문과 답변으로 구성되어 있습니다|간단한 일상 대화, 감정 공감에 집중한 데이터를 가져왔습니다.|
|--|--|--|
|1|가상화폐 쫄딱 망함|어서 잊고 새출발하세요.|
|2|가스비가 너무 많이 나옴|다음 달에는 더 절약해봐요|
|3|눈 앞이 깜깜할 때도 있어요| 그랬군요. 제가 당신 마음을 조금 더 이해할 수 있으면 좋겠어요.|

### 데이터 전처리

<img width="900" alt="image" src="https://user-images.githubusercontent.com/87808408/163403596-eb3a5762-f8bc-4465-9194-d5d182ce506f.png">

데이터 전처리 과정이 가장 중요했습니다. 문장부호 등을 제거한 전처리한 대화집을 txt 형태로 바꾸어 토큰화 한 후 1번 이상 나온 단어에 대해서 단어집을 만들고, 그 단어에 인덱스를 부여해 토큰화된 문장에 인덱스를 부여했습니다. 인덱스 부여 후 이를 텐서로 변환해서 이용했습니다. 이때 seq2seq은 문장의 입력 길이가 같아야하기에 padding 을 추가해주어야하고 문장의 시작과 끝을 명시해야하기 때문에 eos token, sos token, pad token 등 특수 토큰을 단어집에도 추가하고 문장에도 추가해 주었습니다. <br>

---

### MODEL

[참고논문_Sequence To Sequence Learning With Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) <br>
[참고논문_Attention Is All You Need](https://arxiv.org/abs/1706.03762) <br>
위 논문에서 나오는 사진과 자료 이용했습니다. 문제시 삭제하겠습니다. <br>

<img width="900" alt="image" src="https://user-images.githubusercontent.com/87808408/163405923-778c5a7e-f4a2-4cf7-b716-a6ea5fd62094.png">

Encoder ⇨ 순차적으로 입력 받은 단어정보를 압축해서 context vector 생성 ⇨ Decoder ⇨ 단어를 한 개씩 출력, end token 이 나오기 전까지 수행

기본 모델로 seq2seq을 이용하고 이후 결과를 보고 global attention 을 추가로 구현했습니다.  

### MODEL 특이점

1. 양방향 GRU 를 사용
2. Global attention 이용
3. Global attention 의 계산 방법을 세가지 구현

