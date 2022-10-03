## Machine Learning

**머신러닝**, 세상에 일어나는 일을 설명하고 예측하기 위해 **모델** 또는 **모형** 제작하는 많은 방법 중 하나
=> 데이터를 기반으로 학습을 통해 적합한 모델을 찾아가는 방식

기계가 많은 데이터를 사용해서 결과를 설명할 수 있는 특징, 패턴, 수식을 찾아내도록 하는 것

- 다양한 기법이 존재하고 대개 통계적 기법 및 아이디어를 기반으로 함

- 이 기법들을 구현하는 과정에서 알고리즘, 컴퓨터과학, 수학이 어우러져 있음

- 이미 많은 툴들이 개발되어 있고 코드 사용은 쉽지만
  코드가 담고있는 배경과 이론 파악이 쉽지않다

- ``Supervised Learning (지도학습)``
  
  - **_Infering unknowns from knowns_**
  - 데이터의 정답에 해당하는 레이블을 학습해 새로운 데이터의 레이블 예측 => 일반화
  - 일반화 성능이 제일 좋은 모델을 찾는 문제
  - **K-최근접 이웃** [(KNN 정리 및 구현)](https://github.com/plibi/Machine-Learning/blob/master/KNN.ipynb)
  - **Decision Tree** [(DT 정리 및 구현)](https://github.com/plibi/Machine-Learning/blob/master/DecisionTree.ipynb)
  - **SVM** [(SVM 구현)](https://github.com/plibi/Machine-Learning/blob/master/SVM.ipynb)
  - **Ensemble**
  
  

- ``Unsupervised Learning (비지도학습)``

  - 데이터에 내재된 패턴 찾기
  - **K-Means**
  - **DBSCAN**
  - ...

- ``Reinforcement Learning (강화학습)``

  - 연속된 최적의 의사결정방법 찾기




#### Hyperparameter Tuning

- Hyperparameter를 수정해가며 모델을 최적화하는 과정
- 그리드서치, 랜덤 서치 등등
- 간단한 ML 모델 튜닝 [(코드)](https://github.com/plibi/Machine-Learning/blob/master/HyperparameterTuning.ipynb)
- 딥러닝 모델 튜닝[(코드)](https://github.com/plibi/Machine-Learning/blob/master/DNN%20Hyperparameter%20Tuning.ipynb)




## DeepLearning

- ``MLP``
  
  - Multi-Layer Perceptron, 

- ``CNN``
  
  - Convolutional Neural Network, [(코드)](https://github.com/plibi/Machine-Learning/blob/master/Deep%20Learning/CNN.ipynb)

- ``RNN``
  
  - Recurrent Neural Network, 
  
  

## NLP, Natural Language Processing

- ``Word Embedding``
- ``Text Classification``
- ``Sentiment Analysis``
- ``Text Similarity``
- ``Chatbot``
- ``Machine Translation``



## Case Study Project

1. Titanic Survivor Prediction [(Kaggle)](https://www.kaggle.com/c/titanic)
2. Diabete Patient Prediction [(Kaggle)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
3. Santander Customer Transaction Prediction [(Kaggle)](https://www.kaggle.com/c/santander-customer-transaction-prediction/) - LGBM
4. House Prices Prediction [(Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)
5. Bike Sharing Demand Prediction [(Kaggle)](https://www.kaggle.com/c/bike-sharing-demand/) - linear regression, MLP
6. MovieLens Latest Datasets - Collaborative filtering!
7. Book Recommendation Dataset [(Kaggle)](https://www.kaggle.com/arashnic/book-recommendation-dataset) - Collaborative filtering!
8. Adult Census Income Prediction [(Kaggle)](https://www.kaggle.com/uciml/adult-census-income)
9. ...
10. Bag of Words Meets Bags of Popcorn [(Kaggle)](https://www.kaggle.com/c/word2vec-nlp-tutorial/data) - NLP
11. Naver sentiment movie corpus [(Github)](https://github.com/e9t/nsmc)
12. Quora Question Pairs [(Kaggle)]()
