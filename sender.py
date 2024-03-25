import paho.mqtt.client as mqtt
import base64

# 유겸희 컴퓨터 아이피 주소 127.0.0.1

# MQTT 브로커에 연결됐을 때 호출되는 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

# MQTT 클라이언트 객체 생성 및 콜백 함수 설정
client = mqtt.Client()
client.on_connect = on_connect

# MQTT 브로커에 연결
client.connect("127.0.0.1", 1883, 60)

# 이미지 파일을 바이트로 읽어 Base64 인코딩
with open("image.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# 인코딩된 이미지를 MQTT 메시지로 발행
client.publish("topic/image", image_base64)

# 연결 종료
client.disconnect()
