# fp_manager.py

import numpy as np
from config import *
from utilities import iou, expand_and_clamp, calculate_avg_box, compare_boxes


class FpManager:
    """
    YOLO 탐지 결과에 대해 Static/Dynamic FP 필터링 및 쿨다운을 관리하고
    최종 정탐(TP)을 결정하는 핵심 로직을 캡슐화한 클래스입니다.
    """

    def __init__(self, W, H, fps, mqtt_publisher):
        # 시스템 및 상태 초기화
        self.W, self.H, self.fps = W, H, fps
        self.mqtt_publisher = mqtt_publisher

        # FP 상태 변수: 확정 FP, FP 후보, 쿨다운 영역
        self.fp_candidates = {}
        self.false_positive_regions = []
        self.fp_recently_cleared_regions = []

        # Dynamic FP 분석 변수: 베이스라인 측정 관련
        self.baseline_started = False
        self.base_start_frame = 0
        self.baseline_boxes = []
        self.base_box_avg = None
        self.current_interval_boxes = []

    # ==========================================================================
    # 1. FP 라이프사이클 관리 (쿨다운 로직)
    # ==========================================================================
    def _manage_fp_lifespan(self, frame_count):
        """만료된 FP (마스킹)를 쿨다운으로 이동시키고, 만료된 쿨다운 영역을 제거합니다."""
        newly_cleared = []
        active_regions = []

        # (1) 만료된 FP를 쿨다운으로 이동
        for fp in self.false_positive_regions:
            if fp['expire_at'] > frame_count:
                active_regions.append(fp)
            else:
                newly_cleared.append({
                    'box': fp['box'],
                    'type': fp['type'],
                    'cooldown_expire_at': frame_count + FP_COOLDOWN_FRAMES
                })

        self.false_positive_regions = active_regions
        self.fp_recently_cleared_regions.extend(newly_cleared)

        # (2) 만료된 쿨다운 영역 제거
        self.fp_recently_cleared_regions = [
            fp for fp in self.fp_recently_cleared_regions
            if fp['cooldown_expire_at'] > frame_count
        ]

    # ==========================================================================
    # 2. Static FP 추적 및 확정
    # ==========================================================================
    def _track_static_boxes(self, detections, current_frame_id):
        """지속적으로 나타나는 탐지를 Static FP 후보로 추적하고 확정합니다."""
        new_fp_candidates = {}
        newly_confirmed = []

        # (1) 박스 추적 카운트 갱신
        for b_box, _ in detections:
            matched_key = next((p_box_tuple for p_box_tuple in self.fp_candidates if
                                iou(b_box, np.array(p_box_tuple)) > TRACK_IOU_THRESH), None)
            current_box_tuple = tuple(b_box.astype(float))
            key_to_use = matched_key if matched_key is not None else current_box_tuple
            new_fp_candidates[key_to_use] = self.fp_candidates.get(key_to_use, 0) + 1

        # (2) Static FP 확정 (HOLD_FRAMES 초과 시)
        for b_box_tuple, cnt in new_fp_candidates.items():
            if cnt >= HOLD_FRAMES:
                b_box = np.array(b_box_tuple)
                # 기존 확정 FP와 겹치지 않을 때만 새로 확정
                if all(iou(b_box, fp['box']) < 0.9 for fp in self.false_positive_regions):
                    fp_box = expand_and_clamp(list(b_box), 0.08, self.W, self.H)

                    # 1. FP 확정 정보 저장
                    newly_confirmed.append({
                        'box': fp_box,
                        'expire_at': current_frame_id + FP_LIFESPAN_FRAMES,
                        'type': 'Static'
                    })

                    # 2. Static FP 확정 MQTT 메시지 발행 (추가된 기능)
                    self.mqtt_publisher.publish_alert(
                        location="Static_Noise_Area",
                        status="FALSE_POSITIVE",
                        confidence=0.9,
                        box_coords=list(b_box_tuple),
                        frame_count=current_frame_id,
                        filter_type="Static"
                    )

                    # 디버깅 출력: 마스킹 시간 표시
                    print(f"FP 확정 [Static, Frame {current_frame_id}]: {FP_LIFESPAN_FRAMES / self.fps:.1f}초 마스킹 시작")

        self.false_positive_regions.extend(newly_confirmed)
        self.fp_candidates = {k: v for k, v in new_fp_candidates.items() if v < HOLD_FRAMES}
        return newly_confirmed

    # ==========================================================================
    # 3. Dynamic FP 분석 및 확정
    # ==========================================================================
    def _analyze_dynamic_boxes(self, filtered_boxes, frame_count):
        """탐지 영역의 동적 거동(움직임 변화)을 분석하여 Dynamic FP를 확정합니다."""
        if not filtered_boxes: return

        if not self.baseline_started:
            self.baseline_started = True
            self.base_start_frame = frame_count

        # 1. 베이스라인 수집
        if frame_count < self.base_start_frame + BASELINE_COLLECT_FRAMES:
            self.baseline_boxes.extend(filtered_boxes)
            return

        if self.base_box_avg is None and len(self.baseline_boxes) > 0:
            self.base_box_avg = calculate_avg_box(self.baseline_boxes)
            return

        # 2. 동적 비교 및 확정
        if self.base_box_avg is not None:
            self.current_interval_boxes.extend(filtered_boxes)

            if frame_count % DYNAMIC_COMPARE_INTERVAL == 0 and self.current_interval_boxes:
                curr_box_avg = calculate_avg_box(self.current_interval_boxes)
                result, box_to_mark = compare_boxes(self.base_box_avg, curr_box_avg, self.W, self.H, MOVE_THRESH_RATIO)

                if result == "False Positive (Dynamic)" and box_to_mark is not None:
                    fp_box = expand_and_clamp(list(box_to_mark), 0.08, self.W, self.H)

                    # FP 확정 정보 저장
                    self.false_positive_regions.append({
                        'box': fp_box,
                        'expire_at': frame_count + DYNAMIC_FP_LIFESPAN,
                        'type': 'Dynamic'
                    })

                    # Dynamic FP 확정 MQTT 메시지 발행 (기존 기능)
                    print(f"FP 확정 [Dynamic, Frame {frame_count}]: {DYNAMIC_FP_LIFESPAN / self.fps:.1f}초 마스킹")
                    self.mqtt_publisher.publish_alert(
                        location="Dynamic_Noise_Area", status="FALSE_POSITIVE", confidence=0.8,
                        box_coords=list(box_to_mark), frame_count=frame_count, filter_type="Dynamic"
                    )
                self.current_interval_boxes = []

    # ==========================================================================
    # 4. 최종 정탐 (True Positive) 결정
    # ==========================================================================
    def _determine_true_positive(self, filtered_detections, newly_confirmed):
        """Static 후보, 쿨다운, 신규 확정 FP 영역을 피해 최종 정탐을 결정합니다."""
        final_true_detections = []

        fp_candidate_boxes = [np.array(b_box_tuple) for b_box_tuple in self.fp_candidates.keys()]  # 방어 2: Static 후보
        fp_cooldown_boxes = [fp['box'] for fp in self.fp_recently_cleared_regions]  # 방어 3: 쿨다운 영역
        newly_confirmed_boxes = [fp['box'] for fp in newly_confirmed]  # 방어 4: 신규 확정 FP

        for box, conf in filtered_detections:
            is_excluded = False

            if any(iou(box, fp_candidate_box) > TRACK_IOU_THRESH for fp_candidate_box in fp_candidate_boxes):
                is_excluded = True
            elif any(iou(box, fp_cooldown_box) > FP_MASK_IOU_STRICT for fp_cooldown_box in fp_cooldown_boxes):
                is_excluded = True
            elif any(iou(box, newly_confirmed_box) > FP_MASK_IOU_STRICT for newly_confirmed_box in
                     newly_confirmed_boxes):
                is_excluded = True

            if not is_excluded:
                final_true_detections.append((box, conf))

        return final_true_detections

    # ==========================================================================
    # 5. 메인 실행 메서드
    # ==========================================================================
    def process_frame(self, all_fire_detections, frame_count):
        """한 프레임에 대한 전체 FP 필터 파이프라인을 실행하고 결과를 반환합니다."""

        self._manage_fp_lifespan(frame_count)

        # 1. 1차 필터링: 현재 확정된 FP 영역 제외 (방어 1)
        filtered_detections = [
            (box, conf) for box, conf in all_fire_detections
            if all(iou(box, fp['box']) < FP_MASK_IOU_STRICT for fp in self.false_positive_regions)
        ]

        # 2. Static FP 추적 및 확정
        newly_confirmed = self._track_static_boxes(filtered_detections, frame_count)

        # 3. Dynamic FP 분석 및 확정
        self._analyze_dynamic_boxes([box for box, conf in filtered_detections], frame_count)

        # 4. 최종 정탐 결정 (방어 2, 3, 4 적용)
        final_true_detections = self._determine_true_positive(filtered_detections, newly_confirmed)

        return final_true_detections, self.false_positive_regions, self.fp_recently_cleared_regions