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
- 본 과제의 특징은 테스트 데이터 셋은 예술작품(이미지)의 일부분(약1/4)만을 제공하기 때문에 train과 test 사이의 domain gap 발생
- 이를 해결 하기 위해서 train시에 이미지를 가로,세로를 1/2 비율로 Random Crop하도록 augmentation 진행
- 추가적으로 Overfitting을 방지하고 데이터 diversity를 늘리기 위해서, CutMix, CutOut과 같은 data augmentation 기법 적용
- 모델로는 CNN과 Transformer를 결합한 ConvNext_Large 모델 사용
- 최종적으로 F1-score 0.85487 달성

## 4. 한계점
<!-- Write Overview about this project -->
- train 데이터에 존재하는 화가의 작품 개수가 불균형 해서 data imbalance의 문제가 있었음. 이 부분을 해소하기위한 방법이 부족했음
- 모든 화가의 작품을 균등하게 학습할 수 있도록 Weighted Random Sampling 진행하면 더 좋은 성능을 얻을 것으로 보임

## Team member
장종환 (개인 참가)
