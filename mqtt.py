import paho.mqtt.client as mqtt
import base64
import numpy as np
import io
from PIL import Image

class MQTTClient:
    def __init__(self, broker_address = "192.168.0.41", port = 1883):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        # self.client.on_message = self.on_message
        self.broker_address = broker_address
        self.port = port
    
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        self.client.subscribe("topic/image")

    def on_message(self):
        print("Image received")
        image_bytes = base64.b64decode(msg.payload)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        return image_np

    def connect(self):
        self.client.connect(self.broker_address, self.port, 60)

        self.client.loop_start()

    def publish_image(self, image_np, topic = "topic/image"):
        image = Image.fromarray(image_np)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        # Base64 인코딩하여 MQTT 메시지로 발행
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        self.client.publish(topic, image_base64)
        print("Published image to topic", topic)

    def disconnect(self):
        self.client.disconnect()
        print("Disconnected from MQTT broker")

