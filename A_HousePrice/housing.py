'''
Created on 2022. 4. 19.

@author: pc360
'''
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from time import time
import logging
import os

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MY_EPOCH = 500 # 반복횟수
MY_BATCH = 64 # 일괄처리량

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
heading = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
           'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
           'LSTAT', 'MEDV']

raw = pd.read_csv('housing.csv')

# 데이터 원본 출력
print('원본 데이터 샘플 10개')
print(raw.head(10))

print('원본 데이터 통계')
print(raw.describe())

# Z-점수 정규화
# 결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
Z_data = scaler.fit_transform(raw)

# numpy에서 pandas로 전환
# header 정보 복구 필요
Z_data = pd.DataFrame(Z_data,
                      columns=heading)


# 정규화 된 데이터 출력
print('정규화 된 데이터 샘플 10개')
print(Z_data.head(10))

print('정규화 된 데이터 통계')
print(Z_data.describe())


# 데이터를 입력과 출력으로 분리
print()
print('분리 전 데이터 모양: ', Z_data.shape)
X_data = Z_data.drop('MEDV', axis=1)
Y_data = Z_data['MEDV']

# 데이터를 학습용과 평가용으로 분리
X_train, X_test, Y_train, Y_test = \
    train_test_split(X_data,
                     Y_data,
                     test_size=0.3)

print()
print('학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)


# 상자 그림 출력
sns.set(font_scale=0.8)
sns.boxplot(data=Z_data, palette='dark')
plt.tight_layout()

if os.path.isfile('housingBP.png'):
    os.remove('housingBP.png')
plt.show()
plt.savefig('housingBP.png', bbox_inches='tight')


########## 인공 신경망 구현 ##########

# 케라스 DNN 구현
model = Sequential()
indim = X_train.shape[1]
model.add(Dense(200,
                input_dim=indim,
                activation='relu'))

model.add(Dense(1000,
                activation='relu'))

model.add(Dense(1))

print()
print('DNN 요약')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if os.path.isfile('model.log'):
    os.remove('model.log')
file_handler = logging.FileHandler('model.log')
logger.addHandler(file_handler)
model.summary(print_fn=logger.info)
model.summary()


########## 인공 신경망 학습 ##########

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='sgd',
              loss='mse')

print()
print('DNN 학습 시작')
begin = time()

model.fit(X_train,
          Y_train,
          epochs=MY_EPOCH,
          batch_size=MY_BATCH,
          verbose=1)

end = time()
print('총 학습 시간: {:.1f}초'.format(end - begin))

########## 인공 신경망 평가 및 활용 ##########

# 신경망 평가 및 손실값 계산
loss = model.evaluate(X_test,
                      Y_test,
                      verbose=0)

print()
print('DNN 평균 제곱 오차 (MSE): {:.2f}'.format(loss))


# 신경망 활용 및 산포도 출력
pred = model.predict(X_test)
sns.regplot(x=Y_test, y=pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
