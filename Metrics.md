## Classification Metrics

안외워져서 정리하는 분류 평가지표들

[Dacon포스트](https://dacon.io/en/forum/405817)를 참고해서 정리

- Confusion Matrix

  - |           |    실제 참     |   실제 거짓    |
    | :-------: | :------------: | :------------: |
    |  예측 참  | True Positive  | False Positive |
    | 예측 거짓 | False Negative | True Negative  |

    (정답)(예측값)의 조합



### Accuracy

- 참을 참으로 거짓을 거짓으로 잘 분류한 비율

- $$
  Accuracy = \frac{TP + TN}{TP + FP + FN + TN}
  $$



### Precision

- **참으로 분류한 결과 중**에서 실제 참의 비율

- $$
  Precision = \frac{TP}{TP + NP}
  $$

  (Precision이니까 'p'만 들어간다고 외운다...)



### Recall

- **실제 참 중**에서 참으로 분류한 비율

- $$
  Recall = \frac{TP}{TP + FN}
  $$

  (Precision은 Classifier가 참으로 분류한 것 기준, Recall은 실제 참이 기준)



### F1-score

- Precision과 Recall의 조화평균

- 데이터의 label이 불균형일 때 사용하면 좋다

- $$
  F_1 = 2*\frac{Precision * Recall}{Precision + Recall}
  $$

  