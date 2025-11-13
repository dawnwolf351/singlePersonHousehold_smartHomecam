# -*- coding: utf-8 -*-
"""
================================================================================
MQTT_PUBLISHER.PY: AI 감지 이벤트를 MQTT 브로커에 비동기 발행하는 모듈
================================================================================
"""
import paho.mqtt.client as mqtt
import json
import time
from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_ALERT_TOPIC


class MqttPublisher:
    """
    MQTT 발행자 클라이언트. 백그라운드 스레드를 사용하여 메인 AI 추론 루프와 독립적으로 알림을 전송합니다.
    """

    def __init__(self, client_id="AI_Detector_Publisher"):
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        """브로커 연결 완료/실패 시 호출되는 콜백입니다."""
        if rc == 0:
            print(f"MQTT PUB 브로커 연결 성공: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
            self.connected = True
        else:
            print(f"MQTT 연결 실패, 코드: {rc}")
            self.connected = False

    def start(self):
        """클라이언트를 시작하고 백그라운드 스레드에서 MQTT 루프를 실행합니다."""
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"MQTT 클라이언트 시작 오류: {e}")
            self.connected = False

    def stop(self):
        """클라이언트 루프와 연결을 안전하게 종료합니다."""
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT 발행자 클라이언트 종료됨.")

    def _publish_message(self, status, location, confidence, box_coords, frame_count, filter_type=None, extra_log=""):
        """실제 MQTT 메시지를 구성하고 발행하는 내부 헬퍼 메서드입니다."""
        if not self.connected:
            print("경고: MQTT 브로커와 연결되지 않아 알림 발행 실패.")
            return

        try:
            alert_message = {
                "event_id": frame_count,
                "timestamp": time.time(),
                "location": location,
                "confidence": float(confidence),
                "box_coords": [float(c) for c in box_coords],
                "status": status,
                "filter_type": filter_type if filter_type else "None"
            }

            self.client.publish(
                topic=MQTT_ALERT_TOPIC,
                payload=json.dumps(alert_message),
                qos=1
            )
            print(f"[{status}] Frame {frame_count}: MQTT 발행 성공!{extra_log}")
        except Exception as e:
            print(f"MQTT 발행 실패: {e} (Status: {status})")

    def publish_alert(self, location, status, confidence, box_coords, frame_count, filter_type=None):
        """일반 감지 이벤트 (주로 True Positive)를 MQTT 브로커에 발행합니다."""
        self._publish_message(
            status=status,
            location=location,
            confidence=confidence,
            box_coords=box_coords,
            frame_count=frame_count,
            filter_type=filter_type
        )

    def publish_static_fp_alert(self, location, box_coords, frame_count, filter_type):
        """정적 오탐(Static FP) 확정 이벤트를 MQTT 브로커에 발행합니다."""
        # Static FP 확정은 상태와 신뢰도가 고정되어 있습니다.
        status = "FALSE_POSITIVE"
        confidence = 0.9

        self._publish_message(
            status=status,
            location=location,
            confidence=confidence,
            box_coords=box_coords,
            frame_count=frame_count,
            filter_type=filter_type,
            extra_log=" (Static FP)"
        )