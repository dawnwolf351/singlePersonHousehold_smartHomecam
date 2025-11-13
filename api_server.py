# -*- coding: utf-8 -*-
"""
================================================================================
API_SERVER.PY: MongoDB 로그를 Flutter 앱에 제공하는 Flask API 서버
================================================================================
"""
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
import json
from bson import ObjectId
from datetime import datetime

# --- 1. Flask 앱 및 MongoDB 초기화 ---

app = Flask(__name__)
# 모든 외부 도메인에서의 API 요청을 허용 (Flutter 앱 통신용)
CORS(app)

# MongoDB 클라이언트 연결 시도
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print(f"INFO: API 서버, MongoDB 연결 성공. DB: '{DB_NAME}'")
except Exception as e:
    print(f"ERROR: API 서버, MongoDB 연결 실패: {e}")
    mongo_client = None


# --- 2. 커스텀 JSON 인코더 ---

def custom_json_encoder(obj):
    """
    JSON으로 변환 불가능한 MongoDB 타입(ObjectId, datetime)을
    문자열(str)로 변환합니다.
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()  # ISO 8601 형식 문자열로 변환
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --- 3. API 엔드포인트(라우트) 정의 ---

@app.route('/')
def home():
    """서버 상태 확인(Health Check)을 위한 기본 경로입니다."""
    return jsonify({"status": "Flask API Server is running!"})


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """
    /api/logs (GET 요청)
    MongoDB에서 최근 감지 로그를 조회하여 JSON으로 반환합니다.
    """
    if not mongo_client:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # 1. URL 쿼리 파라미터에서 'limit' 값을 가져옴 (기본값: 10)
        limit = int(request.args.get('limit', 10))

        # 2. MongoDB에서 데이터 조회 (최신순으로 'limit' 개수만큼)
        logs = list(collection.find().sort("timestamp", -1).limit(limit))

        # 3. 커스텀 인코더를 사용하여 JSON 응답을 효율적으로 구성
        response_data = json.dumps(logs, default=custom_json_encoder)

        # Response 객체를 사용하여 JSON 응답을 명시적으로 반환
        return Response(response_data, mimetype='application/json')

    except Exception as e:
        print(f"ERROR: /api/logs 처리 중 오류: {e}")
        return jsonify({"error": str(e)}), 500


# --- 4. Flask 서버 실행 ---

if __name__ == '__main__':
    # host='0.0.0.0': 외부 네트워크 접근을 허용합니다.
    # debug=False: 운영 환경을 고려하여 디버그 모드를 비활성화했습니다.
    app.run(host='0.0.0.0', port=5000, debug=False)