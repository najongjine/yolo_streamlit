from ultralytics import YOLO

# 1. 모델 로드 (경로는 실제 파일 위치로 변경)
model = YOLO("yolo11n_sz640_lr0.001_mos0.0.pt")

print(f"model names: ",model.names)  # 여기 출력되는 클래스 목록이 data.yaml과 동일한지 확인

# 2. 추론 실행 (source에 이미지 경로 입력)
results = model.predict(
    source="C:/Users/itg/Pictures/corn_leaf/train/Common_Rust/Corn_Common_Rust (1296).jpg",    # 테스트할 이미지 경로
    imgsz=640,            # 추론 해상도
    conf=0.5,            # confidence threshold
    device="cpu"              # GPU=0, CPU는 "cpu"
)

# 3. 결과 확인
for result in results:
    print(f"result.names:",result.names)   # 클래스 이름 딕셔너리
    print(f"result.boxes:",result.boxes)   # 감지된 박스 정보
    result.show()         # 창에 시각화
    # result.save(filename="output.jpg")  # 저장도 가능
