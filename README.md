# 개인 프로젝트 - 플라스크를 이용한 단순 기계학습 수행 도커 이미지
- 작업 인원 1명
---
# 차례
- [프로젝트 개요]([https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94](https://github.com/khdbsfdk/flask-image/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
- [프로젝트 내 Tool]([https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C](https://github.com/khdbsfdk/flask-image/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EB%82%B4-tool))
- [app.py]([https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C](https://github.com/khdbsfdk/flask-image/blob/main/README.md#apppy))
- [index.html]([https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%EC%B1%97%EB%B4%87](https://github.com/khdbsfdk/flask-image/blob/main/README.md#apppy))
- [result.html]([https://github.com/khdbsfdk/Chatbot-Doll/blob/main/README.md#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EC%A0%88%EC%B0%A8---%ED%8C%8C%EC%9D%B4%EC%96%B4-%EB%B2%A0%EC%9D%B4%EC%8A%A4](https://github.com/khdbsfdk/flask-image/blob/main/README.md#resulthtml))
- [Dockerfile](https://github.com/khdbsfdk/flask-image/blob/main/README.md#dockerfile)
- [requirements.txt](https://github.com/khdbsfdk/flask-image/blob/main/README.md#requirementstxt)
- [개선 필요 부분](https://github.com/khdbsfdk/flask-image/blob/main/README.md#requirementstxt)
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
# 프로젝트 내 Tool
### (1) Flask를 사용한 이유
- 파이썬을 활용한 기계학습 코드가 필요했고, 이를 출력할 웹이 필요했습니다.
- 플라스크를 이용한 웹을 통해 csv 파일을 불러오고 "Upload and Process" 버튼을 통해 기계학습을 수행합니다.
### (2) 데이터의 종류
- 본 프로젝트는 전처리가 거의 필요 없는 "iris" 데이터를 기준으로 둔 코드 입니다.
### (3) 모델 종류
- 모델은 LogisticRegression를 사용하였습니다.
### (4) 도커 이미지
- 도커 이미지로 만들어 가볍게 사용하고자 했습니다.
- 차후 K8S Service로 동작시키고자 합니다.

# app.py

``` python
pip install flask pandas sklearn
```
``` python
from flask import Flask, render_template, request
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics         import accuracy_score

app = Flask(__name__)

# 데이터프레임을 생성하고 기계 학습 모델을 훈련하는 함수
def process_data(file):
    # 업로드된 CSV 파일을 데이터프레임으로 읽어옴
    df = pd.read_csv(file)
    
    # 마지막 컬럼 이름 'target'으로 변경
    last_col_name = df.columns[-1]
    df = df.rename(columns={last_col_name : 'target'})

    #데이터 전처리 - 결측치 0으로 대체
    df = df.fillna(0)

    #데이터 전처리 - 레이블 인코딩
    encoder = LabelEncoder()
    labels = encoder.fit(df['target']).transform(df['target'])

    # 전처리 후 데이터 프레임
    encoder_frm = pd.DataFrame({'target':labels})
    df['target'] = encoder_frm['target']

    #데이터 컬럼 중 id는 제거
    for i in df.columns:
        if i =='id' or i == 'Id' or i == 'ID':
            df = df.drop(columns=[i])

    # 데이터 전처리 및 특징 데이터(X)와 대상 데이터(y) 추출
    x = df.drop(columns=['target'])
    y = df['target']
    
    # 데이터프레임을 HTML 표로 변환
    html_table = df.to_html()

    # 기계 학습 모델 훈련
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=43)
    
    # 모델 학습
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # 테스트 데이터로 예측 수행
    predictions = model.predict(x_test)

    #모델의 정확도 출력
    score = round(accuracy_score(y_test, predictions), 2)
    score = score * 100
    # print("예측 정확도는 약 {0}% 입니다.".format(score))
    
    return score, html_table

# Flask root 페이지
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 업로드된 파일을 가져옴
        file = request.files['file']
        
        # 파일이 업로드되었다면 처리
        if file:
            # 데이터프레임 생성 및 기계 학습 모델 훈련
            score, html_table = process_data(file)
            
            # 예측 결과
            # prediction = "모델 훈련 및 예측 완료"
            
            return render_template('result.html', score=score, table=html_table)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

# index.html

``` html
<!DOCTYPE html>
<html>
  <head>
    <title>Main Page</title>
  </head>
  <body>
    <h1>Flask를 이용한 간단한 기계학습 HTML</h1>
    <form action="/" method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required><br><br>
      <input type="submit" value="Upload and Process">
    </form>
  </body>
</html>
```
# result.html

``` html
<!DOCTYPE html>
<html>
  <head>
    <title>Result Page</title>
  </head>
  <body>
    <h1>기계학습 분석 결과</h1>
    <p>예측 정확도는 약 {{ score }}% 입니다.</p>
    
    <!-- 데이터프레임을 표로 표시 -->
    <p>전처리 후 데이터프레임은 아래와 같습니다.</p>
    {{ table|safe }}
  </body>
</html
```

# Dockerfile
``` python
# Base 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Flask 애플리케이션 실행
CMD [ "python", "app.py" ]
```

# requirements.txt
``` python
Flask
pandas
scikit-learn
```

``` python
docker build -t flask-web-image .
docker run -p 5000:5000 flask-web-image
```

<img width="40%" src="[https://user-images.githubusercontent.com/84302953/168939761-5d0a0cef-d83d-42d5-9143-13b6810c63db.png](https://github.com/khdbsfdk/flask-image/assets/84302953/e7306a16-3c7a-47a0-a00f-a2b1b0004f8a](https://github-production-user-asset-6210df.s3.amazonaws.com/84302953/241175156-e7306a16-3c7a-47a0-a00f-a2b1b0004f8a.png)"/>

<img width="40%" src="[https://user-images.githubusercontent.com/84302953/168939761-5d0a0cef-d83d-42d5-9143-13b6810c63db.png](https://github.com/khdbsfdk/flask-image/assets/84302953/e7306a16-3c7a-47a0-a00f-a2b1b0004f8a](https://github.com/khdbsfdk/flask-image/assets/84302953/6e86cc29-f3ee-4d53-a433-c54e5521f8d6](https://github.com/khdbsfdk/flask-image/assets/84302953/6e86cc29-f3ee-4d53-a433-c54e5521f8d6)"/>

# 개선 필요 부분
- 데이터의 특성 파악 후 적절한 전처리 코드 추가 필요
- HTML 수정으로 시인성 개선 필요 
