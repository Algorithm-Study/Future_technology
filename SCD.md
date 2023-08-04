# SCD : A Stacked Carton Dataset for Detection and Segmentation(Feb 2021)

## Abstract
종이 상자 탐지는 물류 자동화 기술에서 매우 중요함. 하지만 연구를 위한 대용량의 데이터셋이 없는 상황이 기술 개발을 제한하고 있어서 SCD라는 새로운 데이터셋을 논문을 통해 공개함.
- 데이터 관련 정보
    1. 소스: 인터넷 + 실제 창고 촬용 이미지
    2. 16,136개의 이미지 및 이미지 내부에 250,000개의 instance mask
- 새로운 detector 설계
    1. BGS(Boundary Guided Supervision module)  
        : 박스 간 경계 정보에 더 집중하게 만듦
    2. OPCL(Offset Prediction between Classification and Localization module)  
        : 분류 과정에서의 imbalance 문제 경감  
        : Localization 퀄리티 향상(AP 3.1% ~ 4.7% 향상)

## Introduction
- 데이터셋 타입 -> 한 이미지에는 하나의 그룹만 존재
    1. LSCD(Live Stacked Carton Dataset)-> 7,735장  
        : 실제 창고에서 수집한 데이터
        : 다양한 이미지(texture) 정보를 포함할 수 있도록 여러 장소에서 촬영(e.g. 물류센터, 도매 시장 등)  
    2. OSCD(Online Stacked Carton Dataset) -> 8,401장
        : 온라인에서 수집한 데이터
    