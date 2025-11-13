# fp_manager.py

import numpy as np
from config import *
from utilities import iou, expand_and_clamp


class FpManager:
    """
    YOLO 탐지 결과에 대해 정적 오탐(Static FP) 필터링 및 쿨다운을 관리하고,
    ByteTrack ID 지속성을 기반으로 최종 정탐(True Positive, TP)을 결정하는 핵심 로직 클래스입니다.
    """

    def __init__(self, W, H, fps, mqtt_publisher):
        # 시스템 및 상태 초기화
        self.W, self.H, self.fps = W, H, fps
        self.mqtt_publisher = mqtt_publisher

        # ByteTrack-MOT 상태 변수: 트랙 ID별 지속 프레임 수
        self.active_track_durations = {}

        # FP 상태 변수: Static FP 후보, 확정 Static FP 영역, 쿨다운(최근 해제) 영역
        self.fp_candidates = {}
        self.false_positive_regions = []  # 확정 Static FP (마스킹)
        self.fp_recently_cleared_regions = []  # FP 쿨다운 영역

    # ==========================================================================
    # 1. FP 라이프사이클 관리 (쿨다운 로직)
    # ==========================================================================
    def _manage_fp_lifespan(self, frame_count):
        """만료된 확정 FP를 쿨다운으로 이동시키고, 만료된 쿨다운 영역을 제거합니다."""
        newly_cleared = []
        active_regions = []

        # (1) 만료된 확정 FP를 쿨다운으로 이동
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
        """지속적으로 나타나는 탐지를 Static FP 후보로 추적하고, 임계값 초과 시 확정 FP로 등록합니다."""
        new_fp_candidates = {}
        newly_confirmed = []

        # (1) 박스 추적 카운트 갱신 (IOU 기반 매칭)
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
                # 기존 확정 FP와 겹치지 않을 때만 새로 확정 (중복 방지)
                if all(iou(b_box, fp['box']) < 0.9 for fp in self.false_positive_regions):
                    # FP 박스 확장 (0.03 비율 유지)
                    fp_box = expand_and_clamp(list(b_box), 0.03, self.W, self.H)

                    # 1. 확정 FP 정보 저장
                    newly_confirmed.append({
                        'box': fp_box,
                        'expire_at': current_frame_id + FP_LIFESPAN_FRAMES,
                        'type': 'Static'
                    })

                    # 2. Static FP 확정 MQTT 메시지 발행
                    self.mqtt_publisher.publish_static_fp_alert(
                        location="Static_Noise_Area",
                        box_coords=list(b_box_tuple),
                        frame_count=current_frame_id,
                        filter_type="Static"
                    )

                    print(f"FP 확정 [Static, Frame {current_frame_id}]: {FP_LIFESPAN_FRAMES / self.fps:.1f}초 마스킹 시작")

        self.false_positive_regions.extend(newly_confirmed)
        # 확정되지 않은 FP 후보만 다음 프레임으로 이월
        self.fp_candidates = {k: v for k, v in new_fp_candidates.items() if v < HOLD_FRAMES}
        return newly_confirmed

    # ==========================================================================
    # 3. Track Duration 관리 및 TP 후보 결정
    # ==========================================================================
    def _update_track_durations_and_determine_tp(self, detections_with_id):
        """ByteTrack ID 지속 시간을 갱신하고, 최소 지속 시간을 만족하는 객체만 TP 후보로 반환합니다."""

        current_frame_ids = set()
        final_tp_candidates = []

        # 1. 현재 프레임의 Track ID 갱신 및 지속 시간 카운트
        for box, conf, track_id in detections_with_id:
            if track_id != -1:  # 추적 ID가 부여된 객체만 처리
                current_frame_ids.add(track_id)

                # 지속 프레임 카운트 증가
                self.active_track_durations[track_id] = self.active_track_durations.get(track_id, 0) + 1

                # 2. TP 임계값 검증: 최소 지속 시간(TRACK_MIN_DURATION_FRAMES) 만족 시 TP 후보로 지정
                if self.active_track_durations[track_id] >= TRACK_MIN_DURATION_FRAMES:
                    final_tp_candidates.append((box, conf, track_id))

        # 3. 끊어진 트랙 정리 (현재 프레임에 없는 ID 제거)
        keys_to_delete = [tid for tid in self.active_track_durations if tid not in current_frame_ids]
        for tid in keys_to_delete:
            del self.active_track_durations[tid]

        return final_tp_candidates  # returns list of (box, conf, id)

    # ==========================================================================
    # 4. 최종 정탐 (True Positive) 결정
    # ==========================================================================
    def _determine_true_positive(self, filtered_detections, newly_confirmed):
        """쿨다운 및 신규 확정 FP 영역을 피해 최종 정탐(TP)을 결정합니다."""
        final_true_detections = []

        # 1. 방어 필터 목록 구성
        fp_cooldown_boxes = [fp['box'] for fp in self.fp_recently_cleared_regions]  # 쿨다운 영역
        newly_confirmed_boxes = [fp['box'] for fp in newly_confirmed]  # 신규 확정 Static FP

        # 2. 최종 필터링 적용
        for box, conf, track_id in filtered_detections:
            is_excluded = False

            # (1) FP 쿨다운 영역과 겹치는지 검사
            if any(iou(box, fp_cooldown_box) > FP_MASK_IOU_STRICT for fp_cooldown_box in fp_cooldown_boxes):
                is_excluded = True
                print(f"DEBUG EXCLUDE: Track ID {track_id} (TP 후보)가 FP 쿨다운 영역과 겹쳐 제외됨 (IoU > {FP_MASK_IOU_STRICT}).")
            # (2) 신규 확정 FP 영역과 겹치는지 검사
            elif any(iou(box, newly_confirmed_box) > FP_MASK_IOU_STRICT for newly_confirmed_box in
                     newly_confirmed_boxes):
                is_excluded = True
                print(f"DEBUG EXCLUDE: Track ID {track_id} (TP 후보)가 신규 확정 FP와 겹쳐 제외됨 (IoU > {FP_MASK_IOU_STRICT}).")

            if not is_excluded:
                final_true_detections.append((box, conf, track_id))

        return final_true_detections  # returns list of (box, conf, id)

    # ==========================================================================
    # 5. 메인 실행 메서드
    # ==========================================================================
    def process_frame(self, all_fire_detections_with_id, frame_count):
        """한 프레임에 대한 전체 FP 필터 파이프라인을 실행하고 결과를 반환합니다."""

        # 1. FP 라이프사이클 관리
        self._manage_fp_lifespan(frame_count)

        # 2. 1차 필터링: 현재 확정된 Static FP 마스킹 영역 제외
        filtered_detections = []
        for box, conf, track_id in all_fire_detections_with_id:
            # 확정된 Static FP 마스킹 영역과 겹치지 않는 경우만 통과
            if all(iou(box, fp['box']) < FP_MASK_IOU_STRICT for fp in self.false_positive_regions):
                filtered_detections.append((box, conf, track_id))

        # 3. Static FP 추적 및 확정: FP 후보를 확정 FP로 전환
        detections_only = [(box, conf) for box, conf, _ in filtered_detections]
        newly_confirmed = self._track_static_boxes(detections_only, frame_count)

        # 4. TP 후보 결정: Track Duration을 만족하는 객체만 선정
        final_tp_candidates = self._update_track_durations_and_determine_tp(filtered_detections)

        # 5. 최종 필터링: 쿨다운 및 신규 확정 FP 검사를 통과한 TP만 최종 확정
        final_true_detections = self._determine_true_positive(final_tp_candidates, newly_confirmed)

        # 최종 TP 결과와 FP/쿨다운 상태를 반환
        return final_true_detections, self.false_positive_regions, self.fp_recently_cleared_regions