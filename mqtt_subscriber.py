# -*- coding: utf-8 -*-
"""
================================================================================
MQTT_SUBSCRIBER.PY: AI 감지 이벤트를 MQTT 브로커에서 비동기로 구독하여 수신하는 모듈
================================================================================
"""
import paho.mqtt.client as mqtt
import json
import time
from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_ALERT_TOPIC  # MQTT 설정 정보 임포트


class MqttSubscriber:
    """
    MQTT 구독자 클라이언트 클래스.
    백그라운드 스레드를 사용하여 브로커로부터 알림을 독립적으로 수신합니다.
    """

    def __init__(self, client_id="AI_Detector_Subscriber"):
        # 1. 클라이언트 인스턴스 생성 (Callback API v2 사용)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

        # 2. 콜백 함수 등록
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False

        print("INFO: MQTT 구독자 초기화 완료.")

    # 수정된 부분: properties 인자 추가 (API v2 필수)
    def _on_connect(self, client, userdata, flags, rc, properties):
        """브로커 연결 완료/실패 시 호출되며, 성공 시 토픽을 구독합니다."""
        if rc == 0:
            print(f"INFO: MQTT SUB 브로커 연결 성공: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
            self.connected = True

            # 연결 성공 시 알림 토픽 구독 (QoS 1)
            result, mid = client.subscribe(MQTT_ALERT_TOPIC, qos=1)
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"INFO: 토픽 '{MQTT_ALERT_TOPIC}' 구독 성공.")
            else:
                print(f"WARN: 토픽 구독 실패, 코드: {result}")
        else:
            print(f"ERROR: MQTT 연결 실패, 코드: {rc}")
            self.connected = False

    def _on_message(self, client, userdata, msg):
        """구독한 토픽에서 메시지를 수신했을 때 호출되어 알림을 처리합니다."""
        try:
            # 수신된 페이로드를 JSON 객체로 파싱
            alert_message = json.loads(msg.payload.decode('utf-8'))

            # --- 알림 처리 로직 ---

            # TODO: 여기에 데이터베이스 저장, 외부 알림 전송 등의 실제 로직 구현

            # 수신된 메시지를 콘솔에 출력
            print("\n--- 수신된 AI 감지 알림 ---")
            print(f"토픽: {msg.topic}")
            print(f"상태: {alert_message.get('status')} (장소: {alert_message.get('location')})")
            print(f"신뢰도: {alert_message.get('confidence'):.2f}, 프레임 ID: {alert_message.get('event_id')}")
            print("------------------------------")

        except json.JSONDecodeError:
            print(f"ERROR: 수신된 페이로드 JSON 디코딩 실패. 페이로드: {msg.payload}")
        except Exception as e:
            print(f"ERROR: 메시지 처리 중 오류 발생: {e}")

    def start(self):
        """클라이언트를 시작하고 백그라운드 스레드에서 MQTT 루프를 실행합니다."""
        try:
            # 브로커에 연결 및 백그라운드 루프 시작 (비동기 처리)
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            self.client.loop_start()
            print("INFO: MQTT 구독자 클라이언트 시작됨.")
        except Exception as e:
            print(f"ERROR: MQTT 구독자 클라이언트 시작 오류: {e}")
            self.connected = False

    def stop(self):
        """클라이언트 루프와 연결을 안전하게 종료합니다."""
        self.client.loop_stop()
        self.client.disconnect()
        print("INFO: MQTT 구독자 클라이언트 종료됨.")


# ==============================================================================
## 테스트를 위한 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # 파일을 단독으로 실행할 때 구독 기능 테스트에 사용됨

    subscriber = MqttSubscriber()
    subscriber.start()

    try:
        # 백그라운드 루프 유지를 위해 메인 스레드를 대기 상태로 유지
        print("\nTEST: 구독자가 실행 중입니다. Ctrl+C를 눌러 종료하세요...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTEST: 사용자에 의해 구독자 종료 요청됨.")
    finally:
        subscriber.stop()