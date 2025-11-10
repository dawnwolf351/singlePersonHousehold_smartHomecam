# -*- coding: utf-8 -*-
"""
================================================================================
CONFIG.PY: 하이퍼파라미터 및 경로 중앙 관리 모듈 (최종)
================================================================================
"""
import torch

# ==============================================================================
# 1. 환경 및 경로 설정
# ==============================================================================
RASPBERRY_PI_IP = "113.198.234.35"
PORT = 8554
MODEL_PATH = r"E:/yu_task/handpose3d-main/resources/yolov5/runs/train/train_yolov8/weights/best.pt"

# 1.2 YOLO 및 장치 설정
CONFIDENCE_THRESHOLD = 0.1
FPS = 30 # 스트림의 예상 FPS (시간 계산 기준)
DEVICE_ARG = 0 if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. FP 필터링 파라미터
# ==============================================================================

# 2.1 Static FP 및 Cooldown 설정
HOLD_FRAMES = 30             # FP 확정 기준: 1.0초 (연속 프레임 수)
FP_LIFESPAN_FRAMES = 300     # FP 마스킹 수명: 10.0초 (300 프레임)
FP_COOLDOWN_FRAMES = 30      # [추가] 쿨다운 수명: 1.0초 (30 프레임)

# 2.2 Dynamic FP 설정
BASELINE_COLLECT_FRAMES = 10 # 최초 기준 수집 프레임 수 (10프레임)
DYNAMIC_COMPARE_INTERVAL = 30 # 비교 대상 수집 주기 (30프레임마다)
DYNAMIC_FP_LIFESPAN = FP_LIFESPAN_FRAMES
MOVE_THRESH_RATIO = 0.02     # 동적 분석을 위한 움직임 임계 비율

# 2.3 IoU 임계값
TRACK_IOU_THRESH = 0.93      # 정적 후보 추적 IoU 임계값
FP_MASK_IOU_STRICT = 0.65    # 확정된 FP 영역에 대한 YOLO 결과 사전 차단 IoU
RENDER_FILTER_IOU = 0.65     # 최종 렌더링 시 FP 영역 필터링 IoU

# ==============================================================================
# 3. MQTT 및 Alert 설정
# ==============================================================================
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_ALERT_TOPIC = "/fire/alert"
ALERT_COOLDOWN_SECONDS = 2.0 # [추가] 정탐 알림 발행 간격 제한

# ==============================================================================
# 4. MongoDB 설정 (구독자 로깅용)
# ==============================================================================
# MongoDB 연결 문자열 (사용자 환경에 맞게 조정 필요)
MONGO_URI = "mongodb://localhost:27017/"
# 데이터베이스 이름 (MongoDB Compass에서 생성한 DB 이름과 일치)
DB_NAME = "fire_system_db"
# 컬렉션 이름 (MongoDB Compass에서 생성한 컬렉션 이름과 일치)
COLLECTION_NAME = "detection_logs"

# ==============================================================================
# 5. 시각화 설정 (렌더링)
# ==============================================================================
BOX_THICKNESS = 2
BOX_COLOR = (255, 0, 0)      # 최종 TP 박스 (파란색 BGR)
FP_COLOR = (0, 128, 0)       # 정적 FP 박스 (초록색 BGR)
DYNAMIC_FP_COLOR = (0, 255, 255) # 동적 FP 박스 (노란색 BGR)
FP_COOLDOWN_COLOR = (150, 150, 150) # 쿨다운 영역 박스 색상 (회색 BGR)