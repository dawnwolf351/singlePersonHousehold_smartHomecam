# -*- coding: utf-8 -*-
"""
================================================================================
CONFIG.PY: 하이퍼파라미터 및 경로 중앙 관리 모듈
================================================================================
"""
import torch

# ==============================================================================
# 1. 환경 및 경로 설정
# ==============================================================================
RASPBERRY_PI_IP = "113.198.234.38"
PORT = 8554
MODEL_PATH = r"E:/yu_task/handpose3d-main/resources/yolov5/runs/train/train_yolov8/weights/best.pt"

# 1.2 YOLO 및 장치 설정
CONFIDENCE_THRESHOLD = 0.1
FPS = 30 # 스트림의 예상 FPS (시간 계산 기준)
DEVICE_ARG = 0 if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. FP 필터링 파라미터
# ==============================================================================

# 2.1 Static FP 및 Cooldown 설정 (정적 오탐 영역 마스킹 관리)
HOLD_FRAMES = 30             # Static FP 확정 기준: 1.0초 (연속 프레임 수)
FP_LIFESPAN_FRAMES = 300     # Static FP 마스킹 수명: 10.0초 (300 프레임)
FP_COOLDOWN_FRAMES = 30      # 쿨다운 수명: 1.0초 (30 프레임)

# --- ByteTrack-MOT 기반 설정 ---
TRACK_MIN_DURATION_SECONDS = 2.0  # 정탐(TP) 확정 최소 추적 지속 시간 (2.0초)
TRACK_MIN_DURATION_FRAMES = int(TRACK_MIN_DURATION_SECONDS * FPS) # 프레임 수 변환

# 2.2 Dynamic FP 설정 (ByteTrack의 Track ID 지속성으로 대체되어 주석 처리)
# BASELINE_COLLECT_FRAMES = 10
# DYNAMIC_COMPARE_INTERVAL = 30
# DYNAMIC_FP_LIFESPAN = FP_LIFESPAN_FRAMES
# MOVE_THRESH_RATIO = 0.02

# 2.3 IoU 임계값
TRACK_IOU_THRESH = 0.93      # Static FP 후보 추적 IoU 임계값
FP_MASK_IOU_STRICT = 0.89    # 확정된 FP 영역에 대한 YOLO 결과 사전 차단 IoU
RENDER_FILTER_IOU = 0.89     # 최종 렌더링 시 FP 영역 필터링 IoU

# ==============================================================================
# 3. MQTT 및 Alert 설정
# ==============================================================================
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_ALERT_TOPIC = "/fire/alert"
ALERT_COOLDOWN_SECONDS = 2.0 # 정탐 알림 발행 간격 제한

# ==============================================================================
# 4. MongoDB 설정 (구독자 로깅용)
# ==============================================================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "fire_system_db"
COLLECTION_NAME = "detection_logs"

# ==============================================================================
# 5. 시각화 설정 (렌더링)
# ==============================================================================
BOX_THICKNESS = 2
BOX_COLOR = (255, 0, 0)      # 최종 TP 박스
FP_COLOR = (0, 128, 0)       # 정적 FP 박스
DYNAMIC_FP_COLOR = (0, 255, 255) # 동적 FP 박스 (렌더링을 위해 유지)
FP_COOLDOWN_COLOR = (150, 150, 150) # 쿨다운 영역 박스 색상