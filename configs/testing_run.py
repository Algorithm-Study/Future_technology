# https://mmdetection.readthedocs.io/en/v2.28.2/2_new_data_model.html

_base_= '/opt/ml/item_box_competition/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'

# # change model's num classes
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=100),
#     )
# )

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = (
  '불닭볶음면 할라피뇨치즈'
, '불닭볶음면 화끈한매운맛'
, '게토레이 레몬향'
, '코카콜라'
, '스낵면'
, '몽실몽실 복숭아 아이스티'
, '웰치스 그레이프맛'
, '내커피색깔은블랙'
, '밀키스 캔'
, '생생우동면'
, '까스활명수'
, '신라면 컵 15개입'
, '깨끗한나라 휴지 2개'
, '기네스 extra stout dark lively'
, '필스너우르켈캔'
, '짜파게티 큰사발면'
, '까르보 불닭볶음면'
, '기린 이치방'
, '탐스제로레몬'
, '상큼달콤 요걸리'
, '후루룩 쌀국수 미역국'
, '청하_연하늘색'
, '청하_진파랑색'
, '볶음 간짬뽕 큰컵'
, '속청 쿨'
, '웅진 초록매실'
, '스위트 아메리카노'
, '말표 맥주 캔'
, '프레첼 갈릭버터맛'
, '솔의눈 pet'
, '알로에베라킹 프리미엄 제로'
, '콘트라베이스 black&shot'
, '짜파구리 매운맛'
, '팔도비빔면 컵'
, '치킨커리 우동'
, '메밀 비빔막국수'
, '게토레이블루'
, '포도 봉봉'
, '칸타타 아이스 스위트 아메리카노'
, '코카콜라제로'
, '크리넥스 안심 키친타월'
, '팔도 김치 도시락'
, '농심 신볶게티'
, '컨디션 lady'
, '하이트 제로 논알콜 캔'
, '미닛메이드 오렌지'
, '닥터유 단백질 초코'
, '종근당 산에는 삼'
, '박카스f'
, '칸타타 카라멜마키아토'
, '비상대책 숙취해소'
, '코카콜라 zero sugar 라임맛'
, '맥콜 제로'
, '코카콜라 오리지널 pet'
, '비락 식혜 pet'
, '콘소메맛 팝콘'
, '초코칩쿠키'
, '팔도 왕뚜껑'
, '칸타타 아이스 블랙커피'
, '남양 맛있는 두유 gt'
, '롯데 칠성 사이다 캔'
, '롯데 수박바 에이드'
, '농심 직화쌀짬뽕 컵'
, '진로 토닉 제로'
, '롤티슈 득템'
, '농심 튀김우동 컵'
, '농심 오징어짬뽕 큰사발면'
, '펩시 콜라 캔'
, '박카스 디카페'
, '크루저 블루베리 병'
, '오뚜기 콕콕콕 스파게티 용기면'
, '곰표 오리지널 나쵸'
, '초코에몽 드링크 ice'
, '마스터 바닐라블랙 pet'
, '월매 쌀막걸리'
, '포카리스웨트 pet'
, '칸타타 스위트 아메리카노 캔'
, '농심 오징어짬뽕 컵'
, '파울라너 바이스비어 캔'
, '오뚜기 컵누들 매콤한 맛'
, '오뚜기 육개장 big'
, '하이트 extra cold 캔'
, '뽀로로 딸기맛 pet'
, '농심 육개장 큰사발면'
, '프레첼 체다치즈맛'
, '서든어택 펑 캔'
, '나무야나무야 클린3겹 24롤'
, '웅진 행복가득세트'
, '참이슬 fresh'
, '하늘보리 pet'
, '맥심 에스프레소 top 마스터 라떼 캔'
, 'tealog 납작 복숭아 아이스티 제로'
, '농심 튀김우동 큰사발면'
, '농심 사리곰탕 큰사발면'
, '팔도 짬뽕 왕뚜껑'
, '오뚜기 진라면 매운맛'
, '오뚜기 콕콕콕 마요짜장볶이'
, '얼큰 유부우동 한그릇'
, '오뚜기 열 참깨라면'
, '오뚜기 콕콕콕 치즈볶이'
)
data = dict(
    train=dict(
        img_prefix='/opt/ml/item_box_competition/data/train/',
        classes=classes,
        ann_file='/opt/ml/item_box_competition/data/train.json'),
    val=dict(
        img_prefix='/opt/ml/item_box_competition/data/validation/',
        classes=classes,
        ann_file='/opt/ml/item_box_competition/data/validation.json'),
    test=dict(
        img_prefix='balloon/val/',
        classes=classes,
        ann_file='balloon/val/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'

runner = dict(max_epochs=40)