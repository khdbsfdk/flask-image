# 개인 프로젝트 - 플라스크를 이용한 단순 기계학습 수행 도커 이미지
- 작업 인원 1명
---
# 차례
- [프로젝트 개요](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
- [프로젝트 수행 절차](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C)
  - [안드로이드](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C)
  - [챗봇](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%B1%97%EB%B4%87)
  - [데이터 베이스](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%ED%8C%8C%EC%9D%B4%EC%96%B4-%EB%B2%A0%EC%9D%B4%EC%8A%A4)
- [개선 사항 및 참고 사이트](https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%EA%B0%9C%EC%84%A0-%ED%95%A0-%EC%82%AC%ED%95%AD)
---
# 프로젝트 개요
### (1)프로젝트 구현 내용
- 플라스크 -> 단순 기계학습 실행 후 결과 출력
- 리눅스 -> 도커 파일을 통한 이미지 생성
### (2)프로젝트 의의
- 로컬 환경에서 웹을 통해 손쉽게 정형 데이터 기계학습 모델 생성이 가능합니다.
- 도커 이미지화 함으로써 가볍고 쉽게 사용 가능합니다.
### (3)개발 환경
- 주피터 노트북
- 리눅스 (RHEL8.7)

---
# 프로젝트 수행 절차 - 안드로이드
- 플라스크를 이용한 웹을 통해 csv 파일을 불러오고 "Upload and Process" 버튼을 통해 기계학습을 수행합니다.


# 프로젝트 수행 절차 - Flask
### (1) Flask를 사용한 이유
- 파이썬을 활용한 기계학습 코드가 필요했고, 이를 출력할 웹이 필요했습니다.
### (2) ㄷ의 종류
- 본 프로젝트는 **일상 대화**에 중점을 두었고, 대화용 챗봇을 목적으로 합니다.
- 추가로 음성으로 대화하는 챗봇입니다.
### (3) 챗봇 데이터
- [송영숙님의 위로 챗봇 데이터에서 가져왔습니다.](https://github.com/songys/Chatbot_data)

### (4) 챗봇 학습 진행
- 제가 사용한 모델은 transformers의 bert입니다.

- 저장한 챗봇 데이터 프레임에 임베딩 값을 구한 후 [코사인 유사도](https://bkshin.tistory.com/entry/NLP-8-%EB%AC%B8%EC%84%9C-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95-%EC%BD%94%EC%82%AC%EC%9D%B8-%EC%9C%A0%EC%82%AC%EB%8F%84)를 구합니다.
<img width="100%" src="https://user-images.githubusercontent.com/84302953/168531538-f7e8360a-67f3-4ee5-93fb-feca2cfdb0d9.png"/>

- 질문 문장을 넣으면 데이터 프레임 중 가장 유사한 질문 문장을 찾아 답변합니다.
<img width="80%" src="https://user-images.githubusercontent.com/84302953/168531771-3e8a6037-1623-4bdd-b616-4444064d7c7d.png"/>

# 프로젝트 수행 절차 - 파이어 베이스
### (1) 안드로이드와 연동
- 안드로이드에서 지원하는 데이터 베이스인 파이어 베이스를 이용했습니다.
- 연동 시 파이어 베이스에서 제공하는 자료와 안드로이드 스튜디오의 내부 코드가 달라 고생했습니다.
- 특히 안드로이드 스튜디오의 버전이 높으면 연동이 불가합니다. 따라서 버전을 20년도 버전으로 바꿔서 진행했습니다.
### (2) 파이썬과 연동
- 파이어 베이스와 연결하기 위해 라이브러리를 설치해줍니다.
- 안드로이드와 연결한 파이어 베이스 프로젝트의 파이썬 전용 json 파일을 가져와야합니다.
``` python
pip3 install firebase_admin
```
``` python
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
 
 #Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('chat-db.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://chat-db-32cde-default-rtdb.firebaseio.com/'
})
```
- 파이어 베이스에서 데이터를 읽고 쓰는 방법입니다.
``` python
# 데이터 읽기
dir2 = db.reference('state/log')
print(dir2.get())
# 데이터 쓰기
dir = db.reference('log')
dir.update({'log':'None'})
dir.update({'question':result_text})
```
- 이제 반복문을 사용하여 안드로이드에서 데이터를 저장하면 챗봇이 답변하게 만들어줍니다.
``` python
while 1:
    #계속 데이터 읽기
    ref = db.reference('log/question')
    ref2 = db.reference('state/log')
    dir = db.reference('log')
    dir2 = db.reference('state')

    #ref.get()는 #json형태로 받아옴
    if ref.get() == 'stop':
        break
    # 무한 반복 중에서도 원할 때만 값 변경 하기
    elif ref2.get() != "None":
        score, result_text, label = return_answer(ref.get())

        if score >= 30:
            print(result_text)
            print("score : ", score)
            dir.update({'question':result_text})
            dir2.update({'log':'None'})
        else:
            print("잘 못 알아들었어요.")
            dir.update({'question':"잘 못 알아들었어요."})
            dir2.update({'log':'None'})
```

<img width="40%" src="https://user-images.githubusercontent.com/84302953/168939761-5d0a0cef-d83d-42d5-9143-13b6810c63db.png"/>

# 개선 필요 부분
- 데이터의 특성 파악 후 적절한 전처리 코드 추가 필요
- HTML 수정으로 시인성 개선 필요 
