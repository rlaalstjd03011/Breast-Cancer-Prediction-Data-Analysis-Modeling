import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


# 1. 데이터 불러오기
data = load_breast_cancer()
x = data.data # 30개 정도의 입력 값 (세포 밀도,크기..etc)
y = data.target # 정답 (암=0, 정상=1)

# 2. 학습용, 테스트용 데이터 분리 (80% 학습, 20% 테스트)
# random_state = 42 : 데이터 나누는 방법을 고정해서 항상 같은 결과!

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000) # max_iter = 10000으로 학습이 잘 되도록 충분히 크게 설정

model.fit(x_train, y_train) # 학습 시작 (가중치 W와 b 조정)
y_pred = model.predict(x_test) # 학습 후 테스트 데이터 예측

accuracy = accuracy_score(y_test, y_pred)
print(f'정확도:{accuracy:.3f}')

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", cm) # 혼동 행렬: 모델이 맞춘것과 틀린것을 보여줌
print(f"Precision: {precision:.3f}") # 정밀도: 암이라고 예측한 것중 진짜 암인 비율
print(f"Recall: {recall:.3f}") # 민감도: 실제 암 중 모델이 맞춘 비율
print(f"F1 Score: {f1:.3f}") # F1 score: Precision과 Recall의 조화 평균

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head()) # 첫 5명의 환자데이터가 보임

# target별 count 그래프 (암/정상 개수 비교)
sns.countplot(x='target', data=df)
plt.title('Number of Patients: Cancer(0) vs. Normal(1)')
plt.xlabel('Patient Status (0 = Cancer, 1 = Normal)')
plt.ylabel('Count')
plt.show()

# 변수 간 관계 시각화: 예를 들어 mean radius와 mean texture
sns.scatterplot(x='mean radius', y='mean texture', hue='target', data=df)
plt.title('mean radius vs mean texture')
plt.show()