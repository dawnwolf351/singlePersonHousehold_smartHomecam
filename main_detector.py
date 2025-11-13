# -*- coding: utf-8 -*-
"""
================================================================================
MAIN_DECTOR.PY: 실시간 하이브리드 오탐지 필터 파이프라인 메인 실행 모듈
================================================================================
"""
# --- Core Libraries ---
import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np

# --- Project Modules ---
from config import *
from mqtt_publisher import MqttPublisher
from fp_manager import FpManager
from utilities import iou

# 전역 변수: 최초 감지 시점 추적
IS_FIRST_DETECTION = True


# ==============================================================================
## 1. 초기화 및 설정
# ==============================================================================
def initialize_system():
    """YOLO 모델, 장치 설정 및 스트림 연결을 처리하고 FP Manager를 초기화합니다."""

    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE_ARG == 0 else "cpu"
    print(f"INFO: 감지된 장치: {DEVICE_NAME} (사용: {DEVICE_ARG})")

    # 1.1 모델 로드
    try:
        model = YOLO(MODEL_PATH)
        device = torch.device(DEVICE_ARG)
        model.to(device)
        print(f"모델 로드 성공. 장치: {model.device}")
    except Exception as e:
        print(f"모델 로드 오류: {e}");
        exit()

    # 1.2 스트림 연결
    stream_url = f"tcp://{RASPBERRY_PI_IP}:{PORT}?video=h264"
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("연결 실패: 스트리밍 상태 및 방화벽을 확인하세요.");
        exit()
    print("연결 성공. 실시간 프레임 수신 및 YOLO 탐지 시작. 'q'를 누르면 종료됩니다.")

    # 1.2.1 FPS 설정 및 비정상 값 처리
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FPS 속성을 재측정하여 초기화를 유도합니다.
    measured_fps = cap.get(cv2.CAP_PROP_FPS)

    # 인식된 FPS 값이 비정상인 경우 config.FPS (30)를 사용합니다.
    if measured_fps <= 1.0 or measured_fps > 1.0e+5:
        fps = FPS  # Fallback: config.py의 FPS 사용
        print(f"DEBUG: 스트림 FPS 인식 실패 ({measured_fps}). config.FPS ({FPS}) 강제 적용.")
    else:
        fps = measured_fps
        print(f"DEBUG: 시스템에서 인식된 FPS: {fps}")

    # 1.3 MQTT 및 FP Manager 초기화
    mqtt_publisher = MqttPublisher()
    mqtt_publisher.start()
    fp_manager = FpManager(W, H, fps, mqtt_publisher)

    return model, cap, fps, mqtt_publisher, fp_manager


# ==============================================================================
## 2. 렌더링 (UI/UX)
# ==============================================================================
def render_frame(frame, final_tp_detections, fp_regions, fp_cooldown_regions, fps, frame_count):
    """프레임에 FP 영역 (확정/쿨다운) 및 최종 TP 박스를 렌더링합니다."""

    # FP 영역 및 쿨다운 영역 렌더링
    for fp_entry in fp_regions + fp_cooldown_regions:
        fp_box = fp_entry['box']
        x1, y1, x2, y2 = map(int, fp_box)
        fp_type = fp_entry['type']
        is_cooldown = 'cooldown_expire_at' in fp_entry

        color = FP_COOLDOWN_COLOR if is_cooldown else (FP_COLOR if fp_type == 'Static' else DYNAMIC_FP_COLOR)

        if is_cooldown:
            # 남은 시간 계산 (쿨다운)
            remaining_frames = max(0, fp_entry['cooldown_expire_at'] - frame_count)
            label = f'{fp_type} Cooldown ({round(remaining_frames / fps, 1)}s)'
        else:
            # 남은 시간 계산 (FP 마스킹)
            remaining_frames = max(0, fp_entry['expire_at'] - frame_count)
            label = f'{fp_type} FP ({round(remaining_frames / fps, 1)}s)'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 최종 정탐(TP) 박스 렌더링
    all_active_fp = fp_regions + fp_cooldown_regions  # IoU 계산을 위해 모든 활성 FP 영역을 모음

    for detection_info in final_tp_detections:
        # 감지 정보 파싱: (box, conf) 또는 (box, conf, id)
        if len(detection_info) == 2:
            box, conf = detection_info
            track_id = None
        else:
            box, conf, track_id = detection_info

        # TP가 활성 FP 영역과 겹치면 렌더링에서 제외 (필터)
        is_masked = False
        for fp_entry in all_active_fp:
            # RENDER_FILTER_IOU 이상 겹치면 마스킹 처리
            if iou(box, fp_entry['box']) > RENDER_FILTER_IOU:
                is_masked = True
                break

        if is_masked:
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # IoU 시각화 정보 생성 (디버깅용)
        max_iou_with_fp = 0.0
        for fp_entry in all_active_fp:
            current_iou = iou(box, fp_entry['box'])
            if current_iou > max_iou_with_fp:
                max_iou_with_fp = current_iou

        id_str = f" ID:{track_id}" if track_id is not None else ""
        iou_str = f" IOU:{max_iou_with_fp:.2f}" if max_iou_with_fp > 0.1 else ""
        label = f'Fire (TP) {conf:.2f}{id_str}{iou_str}'

        # 텍스트 위치: 좌측 하단 바깥쪽
        cv2.putText(frame, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOX_COLOR, 2)

    cv2.imshow('Real-time Video Stream (YOLO + Hybrid FP Filter)', frame)


# ==============================================================================
## 3. 메인 실행 루프
# ==============================================================================
def main_loop(model, cap, fps, mqtt_publisher, fp_manager):
    """실시간 프레임 처리, 필터링, 알림 발행을 반복하는 메인 루프."""
    global IS_FIRST_DETECTION

    frame_count = 0
    start_time = time.time()
    last_alert_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패. 스트림이 끊어졌거나 서버가 종료되었습니다.");
                break

            frame_count += 1
            current_time = time.time()

            # A. YOLO 추론 및 ByteTrack 추적 수행
            results = model.track(
                frame,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=0.5,
                imgsz=640,
                device=model.device,
                tracker='bytetrack.yaml'  # ByteTrack 설정 파일 지정
            )[0]

            # B. 탐지 결과에서 Fire 클래스 및 ByteTrack ID 추출
            all_fire_detections_with_id = []
            track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
            track_ids_list = track_ids.tolist() if len(track_ids) > 0 else []
            id_index = 0

            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                if model.names[int(cls)] == "fire":
                    final_track_id = int(track_ids_list[id_index]) if id_index < len(track_ids_list) else -1
                    all_fire_detections_with_id.append((box.cpu().numpy(), float(conf.cpu().numpy()), final_track_id))
                    id_index += 1

            # 유효한 추적 ID 개수 계산
            valid_track_ids = [tid for _, _, tid in all_fire_detections_with_id if tid != -1]
            current_id_count = len(valid_track_ids)

            # C. FP Manager를 통한 최종 필터링 및 TP 결정
            final_true_detections, fp_regions, fp_cooldown_regions = fp_manager.process_frame(
                all_fire_detections_with_id, frame_count
            )

            # D. 최종 정탐 알림 발행 (시간 쿨다운 적용)
            if final_true_detections and current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
                best_detection = final_true_detections[0]
                best_box, best_conf, best_id = best_detection

                # TP 발행 로그 출력
                print("=" * 60)
                print(f"[TRUE_POSITIVE] Frame {frame_count}: Confirmed Fire Detected!")
                print(f"  > FRAME COUNT: {frame_count}")
                print(f"  > ID COUNT (추적 객체 수): {current_id_count}")
                print(f"  > TRACK ID (발행): {best_id}, Confidence: {best_conf:.3f}")

                # MQTT 발행 (성공 로그는 mqtt_publisher.py에서 출력됨)
                mqtt_publisher.publish_alert(
                    location="Living_Room_Main", status="CONFIRMED_FIRE_DETECTED", confidence=float(best_conf),
                    box_coords=list(best_box), frame_count=frame_count, filter_type="Track_Duration"
                )
                last_alert_time = current_time
                print("=" * 60)

            # E. 렌더링 및 종료 확인
            render_frame(frame, final_true_detections, fp_regions, fp_cooldown_regions, fps, frame_count)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n수동 종료 요청.")
    finally:
        shutdown(start_time, frame_count, mqtt_publisher, cap)


def shutdown(start_time, frame_count, mqtt_publisher, cap):
    """프로그램 종료 시 자원(MQTT, 캡처)을 정리하고 성능 통계를 출력합니다."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    average_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"\n스트림 종료. 총 프레임: {frame_count}")
    print(f"평균 수신 및 처리 속도 (FPS): {average_fps:.2f}")

    mqtt_publisher.stop()
    print("AI 모듈 및 MQTT 클라이언트 종료.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model, cap, fps, mqtt_publisher, fp_manager = initialize_system()
    main_loop(model, cap, fps, mqtt_publisher, fp_manager)