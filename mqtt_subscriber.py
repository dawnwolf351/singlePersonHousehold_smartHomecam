# -*- coding: utf-8 -*-
"""
================================================================================
MQTT_SUBSCRIBER.PY: AI 감지 이벤트를 구독하고 MongoDB에 로그로 저장하는 모듈
================================================================================
"""
import paho.mqtt.client as mqtt
import json
import time
from pymongo import MongoClient
from datetime import datetime, timezone  # BSON UTC datetime 생성을 위해 임포트
from config import (
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_ALERT_TOPIC,
    MONGO_URI, DB_NAME, COLLECTION_NAME  # DB 및 MQTT 설정 임포트
)


class MqttSubscriber:
    """
    MQTT 구독자 클라이언트 클래스.
    비동기 알림 수신 및 MongoDB 로그 저장을 담당합니다.
    """

    def __init__(self, client_id="AI_Detector_Subscriber"):
        # 1. MQTT 클라이언트 초기화 (Callback API v2 사용)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False
        self.mongo_client = None

        # 2. MongoDB 클라이언트 연결
        try:
            self.mongo_client = MongoClient(MONGO_URI)
            self.db = self.mongo_client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print(f"INFO: MongoDB 연결 성공. DB: '{DB_NAME}'")
        except Exception as e:
            print(f"ERROR: MongoDB 연결 실패: {e}")
            self.mongo_client = None

        print("INFO: MQTT 구독자 초기화 완료.")

    def _on_connect(self, client, userdata, flags, rc, properties):
        """브로커 연결 콜백: 성공 시 토픽 구독"""
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
        """메시지 수신 콜백: JSON 파싱 및 MongoDB 저장"""
        try:
            # 1. 수신된 페이로드를 JSON 객체로 파싱
            alert_message = json.loads(msg.payload.decode('utf-8'))

            # 2. MongoDB 저장 로직
            if self.mongo_client:
                # MongoDB Time-Series 컬렉션 요구사항에 맞게 'timestamp' 필드에
                # BSON UTC datetime 객체를 저장 (발행자 메시지에 'timestamp'가 있어도 덮어씀)
                alert_message['timestamp'] = datetime.now(timezone.utc)

                result = self.collection.insert_one(alert_message)
                print(f"INFO: MongoDB 저장 성공. Document ID: {result.inserted_id}")
            else:
                print("WARN: MongoDB 연결 없음. 로그 저장을 건너뜁니다.")

            # 3. 수신 확인을 위한 콘솔 출력
            print("\n--- 수신된 AI 감지 알림 ---")
            print(f"상태: {alert_message.get('status')} (장소: {alert_message.get('location')})")
            print(f"신뢰도: {alert_message.get('confidence'):.2f}, 프레임 ID: {alert_message.get('event_id')}")
            print("------------------------------")

        except json.JSONDecodeError:
            print(f"ERROR: 페이로드 디코딩 실패.")
        except Exception as e:
            print(f"ERROR: 메시지 처리 또는 MongoDB 저장 중 오류 발생: {e}")

    def start(self):
        """MQTT 클라이언트 연결 및 백그라운드 루프 시작"""
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            self.client.loop_start()
            print("INFO: MQTT 구독자 클라이언트 시작됨.")
        except Exception as e:
            print(f"ERROR: MQTT 구독자 클라이언트 시작 오류: {e}")
            self.connected = False

    def stop(self):
        """MQTT 및 MongoDB 연결 안전하게 종료"""
        self.client.loop_stop()
        self.client.disconnect()
        if self.mongo_client:
            self.mongo_client.close()
        print("INFO: MQTT 구독자 클라이언트 종료됨.")


# ==============================================================================
## 독립 실행 시 테스트 로직
# ==============================================================================
if __name__ == '__main__':
    # 파일을 단독으로 실행하여 구독 및 로깅 기능을 테스트합니다.

    subscriber = MqttSubscriber()
    subscriber.start()

    try:
        # 루프 유지를 위해 메인 스레드를 대기 상태로 유지
        print("\nTEST: 구독자가 실행 중입니다. Ctrl+C를 눌러 종료하세요...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTEST: 사용자에 의해 구독자 종료 요청됨.")
    finally:
        subscriber.stop()