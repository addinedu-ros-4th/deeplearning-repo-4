import paho.mqtt.client as mqtt
import base64

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # 이미지를 송신한 토픽 구독
    client.subscribe("topic/image")

def on_message(client, userdata, msg):
    print("Image received")
    # 수신된 메시지에서 Base64 인코딩된 이미지 데이터 추출 및 디코딩
    image_bytes = base64.b64decode(msg.payload)
    # 이미지 파일로 저장
    with open("received_image.jpg", "wb") as image_file:
        image_file.write(image_bytes)

# MQTT 클라이언트 객체 생성 및 콜백 함수 설정
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# MQTT 브로커에 연결
client.connect("127.0.0.1", 1883, 60)

# 네트워크 루프 시작
# 이 루프는 새로운 메시지를 수신하고, 콜백 함수를 처리하는 데 사용됩니다.
client.loop_forever()
