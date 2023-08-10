# Future_technology
미래 기술 챌린지 대회 환경 및 기록  
[대회 공고](https://github.com/Algorithm-Study/Future_technology/blob/main/images/%EB%AF%B8%EB%9E%98%EA%B8%B0%EC%88%A0%EC%B1%8C%EB%A6%B0%EC%A7%80.jpg)

## 과제
**비전을 활용한 스마트 검수**
- Train  
    : 예측해야 하는 클래스 객체가 이미지 하나에 한개씩 존재(약 600장)
- Valid  
    : 예측해야 하는 사물들이 적재되어 있는 Cart 이미지 존재(45장)

- Class
    : 100개의 클래스(오프라인 예선 이전) + 5개의 추가 크래스(예선)

## 예선 진행 시나리오(시나리오를 기반으로 모델 프로세스 파이프라인 작성)
1. SKU110K Cascade RCNN으로 이미지 속 존재하는 상품 객체를 인식해서 Cropped된 데이터셋 생성
2. 분류모델을 활용해서 Cropped된 이미지에 대해 분류 진행
3. 분류 결과를 바탕으로 검출 bbox간 통합 과정을 거침
4. 이후 통합된 bbox에 대해서 GT와의 IOU 계산
5. 조건에 맞게 결과물 출력(bbox 이미지 내부에 그리기 및 txt에 예측 레이블 출력)

## 대회 준비 기록

## 참고 자료

|      Description      | Performance (Accuracy) |
| :-------------------: | :--------------------: |
| LR=0.001 IMGSIZE=224  |           32           |
| LR=0.0005 IMGSIZE=224 |     35 at 45 epoch     |
| LR=0.0005 IMGSIZE=448 |     25 at 46 epoch     |
|                       |                        |