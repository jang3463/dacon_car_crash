# DACON-Car Carsh-AI-Competition
## 1. 개요
https://dacon.io/competitions/official/236064/overview/description
  - 주제 : 블랙박스 영상의 자동차 충돌 분석 AI 알고리즘 개발
  - Task : Video Classification
  - 기간 : 2023.02.06 ~ 2023.03.13
  - 결과 : 24등 / 449
<!--  Other options to write Readme
  - [Deployment](#deployment)
  - [Used or Referenced Projects](Used-or-Referenced-Projects)
-->
## 2. 데이터셋 설명
<!--Wirte one paragraph of project description --> 
![TRAIN_0001](https://github.com/jang3463/dacon_car_crash/assets/70848146/6d42946b-739b-43f8-8d71-eaa4883c8a0f)
- train(폴더) :  학습용 차량 블랙박스 영상, TRAIN_0000.mp4 ~ TRAIN_2697.mp4

- test(폴더) : 평가용 차량 블랙박스 영상, TEST_0000.mp4 ~ TEST_1799.mp4


- train.csv
  - sample_id : 영상 샘플 고유 id
  - video_path : 학습용 차량 블랙박스 영상 경로
  - label : 13가지의 차량 충돌 상황

- test.csv
  - sample_id : 영상 샘플 고유 id
  - video_path : 학습용 차량 블랙박스 영상 경로

<img width="700" height="800" alt="image" src="https://github.com/jang3463/dacon_car_crash/assets/70848146/13c17d46-a5c2-48da-b07b-dab3a9b2cc1d">

## 3. 수행방법
<!-- Write Overview about this project -->
- 본 과제는 블랙박스 영상으로부터 자동차의 충돌 상황을 분석하는 AI 모델을 개발하는 것
- 본 데이터의 LABEL의 위의 이미지처럼 분할이 가능함
- label을 crash, weather, timing으로 분할하여 multi_label_classification 문제로 변환
- 모델로는 slowfast_r101, MVITv2_B_32x3, r3d_18을 사용해본 결과, r3d_18이 성능이 가장 좋았음
- 최종적으로 carsh는 r3d_18 모델 사용, weather, timing은 영상에서 각 5개의 이미지를 random으로 추출하고 convnext_large 모델 사용
- 각 label별로 병렬적으로 임베딩 추출하여 concat
- 최종적으로 F1-score 0.57464 달성

## 4. 한계점
<!-- Write Overview about this project -->
- 데이터 특성상 클래스 불균형도 심하고 잘못 labeling 된 데이터도 포함되어 있어서 쉽지 않은 대회였음

## Team member
장종환 (개인 참가)
