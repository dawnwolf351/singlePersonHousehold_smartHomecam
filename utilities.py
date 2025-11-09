# -*- coding: utf-8 -*-
"""
================================================================================
UTILITIES.PY: 수학적 계산 및 헬퍼 함수 모듈
================================================================================
"""
import numpy as np


# ==============================================================================
# 1. IoU 및 박스 변환
# ==============================================================================
def iou(boxA, boxB):
    """두 바운딩 박스 간의 IoU (Intersection over Union)를 계산합니다."""
    if isinstance(boxA, dict): boxA = boxA['box']
    if isinstance(boxB, dict): boxB = boxB['box']

    xA = max(boxA[0], boxB[0]);
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def expand_and_clamp(box, ratio, W, H):
    """박스를 비율만큼 확장하고 이미지 경계 내로 조정합니다."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio, h * ratio

    nx1 = max(0, x1 - dx);
    ny1 = max(0, y1 - dy)
    nx2 = min(W, x2 + dx);
    ny2 = min(H, y2 + dy)

    return [nx1, ny1, nx2, ny2]


# ==============================================================================
# 2. 동적 분석 헬퍼 함수
# ==============================================================================
def calculate_avg_box(boxes):
    """수집된 박스들의 평균 좌표를 계산합니다."""
    if not boxes:
        return None
    return np.mean(np.array(boxes), axis=0)


def compare_boxes(base_box_avg, curr_box_avg, W, H, move_thresh_ratio):
    """기준 박스와 현재 박스의 평균 좌표를 비교하여 동적 FP 또는 Fire 여부를 판단합니다."""
    if base_box_avg is None or curr_box_avg is None:
        return "No Data", None

    # 이미지 크기 대비 움직임 임계값 설정
    MOVE_THRESH_X = W * move_thresh_ratio
    MOVE_THRESH_Y = H * move_thresh_ratio

    # 주요 경계 변화 계산
    dy_top = abs(curr_box_avg[1] - base_box_avg[1])
    dx_left = abs(curr_box_avg[0] - base_box_avg[0])
    dx_right = abs(curr_box_avg[2] - base_box_avg[2])

    # 바닥 중심 좌표 변화
    curr_center_bottom_x = (curr_box_avg[0] + curr_box_avg[2]) / 2
    base_center_bottom_x = (base_box_avg[0] + base_box_avg[2]) / 2
    dx_center_bottom = abs(curr_center_bottom_x - base_center_bottom_x)

    # 움직임 조건 검사
    move_top = dy_top > MOVE_THRESH_Y
    move_left = dx_left > MOVE_THRESH_X
    move_right = dx_right > MOVE_THRESH_X
    move_center_bottom = dx_center_bottom > MOVE_THRESH_X

    # 판단 로직
    if sum([move_top, move_left, move_right]) >= 2:
        return "Fire", curr_box_avg
    elif move_center_bottom and not (move_top or move_left or move_right):
        return "False Positive (Dynamic)", curr_box_avg
    else:
        return "No Decision", curr_box_avg