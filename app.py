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

# Flask 루트 페이지
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