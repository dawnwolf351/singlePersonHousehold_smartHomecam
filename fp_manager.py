# fp_manager.py

import numpy as np
from config import *
from utilities import iou, expand_and_clamp  # compare_boxes, calculate_avg_box 등은 필요에 따라 제거


class FpManager:
    """
    YOLO 탐지 결과에 대해 Static FP 필터링 및 쿨다운을 관리하고
    ByteTrack ID 지속성을 기반으로 최종 정탐(TP)을 결정하는 핵심 로직을 캡슐화한 클래스입니다.
    """

    def __init__(self, W, H, fps, mqtt_publisher):
        # 시스템 및 상태 초기화
        self.W, self.H, self.fps = W, H, fps
        self.mqtt_publisher = mqtt_publisher

        # 💡 ByteTrack-MOT 상태 변수: {track_id: current_duration_frames}
        self.active_track_durations = {}

        # FP 상태 변수: 확정 FP, FP 후보, 쿨다운 영역
        self.fp_candidates = {}
        self.false_positive_regions = []
        self.fp_recently_cleared_regions = []

        # 🗑️ Dynamic FP 분석 변수 (ByteTrack 대체로 인해 주석 처리)
        # self.baseline_started = False
        # self.base_start_frame = 0
        # self.baseline_boxes = []
        # self.base_box_avg = None
        # self.current_interval_boxes = []

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
    # 2. Static FP 추적 및 확정 (공간적 마스킹 관리를 위해 유지)
    # 💡 Note: process_frame에서 이 함수 호출을 중단하여 Static FP 확정을 막음.
    # ==========================================================================
    def _track_static_boxes(self, detections, current_frame_id):
        """지속적으로 나타나는 탐지를 Static FP 후보로 추적하고 확정합니다."""
        new_fp_candidates = {}
        newly_confirmed = []

        # (1) 박스 추적 카운트 갱신
        for b_box, _ in detections:
            # 기존 IOU 기반 매칭 로직 유지
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

                    # 2. Static FP 확정 MQTT 메시지 발행
                    self.mqtt_publisher.publish_alert(
                        location="Static_Noise_Area",
                        status="FALSE_POSITIVE",
                        confidence=0.9,
                        box_coords=list(b_box_tuple),
                        frame_count=current_frame_id,
                        filter_type="Static"
                    )

                    print(f"FP 확정 [Static, Frame {current_frame_id}]: {FP_LIFESPAN_FRAMES / self.fps:.1f}초 마스킹 시작")

        self.false_positive_regions.extend(newly_confirmed)
        self.fp_candidates = {k: v for k, v in new_fp_candidates.items() if v < HOLD_FRAMES}
        return newly_confirmed

    # ==========================================================================
    # 3. Dynamic FP 분석 및 확정 (ByteTrack의 Track Duration 필터로 대체됨)
    # ==========================================================================
    # def _analyze_dynamic_boxes(self, filtered_boxes, frame_count):
    #     """탐지 영역의 동적 거동(움직임 변화)을 분석하여 Dynamic FP를 확정합니다."""
    #     # 기존 Dynamic FP 로직 전체 주석 처리
    #     pass

    # ==========================================================================
    # 3. Track Duration 관리 및 TP 결정 (ByteTrack ID 기반)
    # ==========================================================================
    def _update_track_durations_and_determine_tp(self, detections_with_id):
        """ByteTrack ID를 기반으로 각 트랙의 지속 시간을 갱신하고 TP를 결정합니다."""

        current_frame_ids = set()
        final_true_detections = []

        # 1. 현재 프레임의 Track ID 갱신 및 지속 시간 카운트
        for box, conf, track_id in detections_with_id:
            if track_id != -1:  # 추적 ID가 부여된 객체만 처리
                current_frame_ids.add(track_id)

                # 지속 프레임 카운트 증가
                self.active_track_durations[track_id] = self.active_track_durations.get(track_id, 0) + 1

                # 2. TP 임계값 검증: config.py의 TRACK_MIN_DURATION_FRAMES 활용
                if self.active_track_durations[track_id] >= TRACK_MIN_DURATION_FRAMES:
                    # 지속 시간을 만족한 객체만 최종 TP 목록에 추가
                    final_true_detections.append((box, conf))

        # 3. 끊어진 트랙 정리 (ByteTrack의 track_buffer가 만료된 트랙 제거)
        keys_to_delete = [tid for tid in self.active_track_durations if tid not in current_frame_ids]
        for tid in keys_to_delete:
            del self.active_track_durations[tid]

        return final_true_detections

    # ==========================================================================
    # 4. 최종 정탐 (True Positive) 결정
    # ==========================================================================
    def _determine_true_positive(self, filtered_detections, newly_confirmed):
        """Static 후보, 쿨다운, 신규 확정 FP 영역을 피해 최종 정탐을 결정합니다."""
        final_true_detections = []

        # (Note: filtered_detections는 이미 Track Duration을 만족한 TP 후보 목록입니다.)

        # 1. 방어 필터 목록 구성
        fp_candidate_boxes = [np.array(b_box_tuple) for b_box_tuple in self.fp_candidates.keys()]  # 방어 2: Static 후보
        fp_cooldown_boxes = [fp['box'] for fp in self.fp_recently_cleared_regions]  # 방어 3: 쿨다운 영역
        newly_confirmed_boxes = [fp['box'] for fp in newly_confirmed]  # 방어 4: 신규 확정 Static FP

        # 2. 최종 필터링 적용
        for box, conf in filtered_detections:
            is_excluded = False

            # Note: 현재 newly_confirmed와 fp_candidates는 process_frame에서
            # _track_static_boxes 호출이 중단되어 항상 비어있을 것입니다.
            # 하지만 안전을 위해 로직은 유지합니다.

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
    # 5. 메인 실행 메서드 (수정됨: Static FP 마스킹 및 추적 비활성화)
    # ==========================================================================
    def process_frame(self, all_fire_detections_with_id, frame_count):
        """한 프레임에 대한 전체 FP 필터 파이프라인을 실행하고 결과를 반환합니다."""

        self._manage_fp_lifespan(frame_count)  # FP 수명 관리 로직은 유지

        # 1. 1차 필터링: 현재 확정된 Static FP 영역 제외 (방어 1)
        # [START] Static FP 마스킹 방어 우회: 모든 감지를 통과
        filtered_detections = all_fire_detections_with_id
        # [END] Static FP 마스킹 방어 우회

        # 2. Static FP 추적 및 확정 (공간적 FP 관리를 위해 유지)
        # ---------------------------------------------------------------------
        # STATIC FP 추적/확정 로직 비활성화 (Track Duration 검증 목적)
        # ---------------------------------------------------------------------
        # detections_only = [(box, conf) for box, conf, _ in filtered_detections]
        # newly_confirmed = self._track_static_boxes(detections_only, frame_count)
        # ---------------------------------------------------------------------

        newly_confirmed = []  # Static FP 확정 없이 빈 목록을 반환

        # 3. Dynamic FP 분석 및 확정 (🗑️ Dynamic FP 로직 호출 제거)

        # 4. 최종 정탐 결정: Track Duration을 만족하는 객체만 TP 후보로 확정
        final_tp_candidates = self._update_track_durations_and_determine_tp(filtered_detections)

        # 5. 최종 필터링: Static 후보 및 쿨다운 영역 검사를 통과한 TP만 최종 확정
        final_true_detections = self._determine_true_positive(final_tp_candidates, newly_confirmed)

        return final_true_detections, self.false_positive_regions, self.fp_recently_cleared_regions