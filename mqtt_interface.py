import os
import paho.mqtt.client as mqtt
import time
import threading

class mqtt_interface():

    def __init__(self):
        self.host = "192.168.29.52"
        self.client = mqtt.Client()
        self.connection = False
        self.last_update = time.time() - 7
        self.client.on_connect = self.connect_cb
        self.client.on_message = self.message_cb
        self.publish_list = []
        self.odom_topic = "odom"
        self.publish_list.append("odom")
        self.imu_topic = "imu"
        self.publish_list.append("imu")
        self.color_topic = "camera/color"
        self.publish_list.append("camera/color")
        self.depth_topic = "camera/depth"
        self.publish_list.append("camera/depth")
        self.segmt_topic = "camera/segmt"
        self.publish_list.append("camera/segmt")
        self.subscribe_topic = "controls"

        proc = threading.Thread(target = self.update_heartbeat)
        proc.setDaemon(True)
        proc.start()

    def connect_cb(self, client, userdata, flags, rc):
        self.connection = True
        for topic in self.topic_list:
            self.client.subscribe(topic)

    def message_cb(self, client, userdata, msg):
        if(msg.topic == self.subscribe_topic):
            self.data_in = msg.payload

    def update_heartbeat(self):
        while True:
            if(time.time() - self.last_update < 6):
                self.RFM_connected = True
            else:
                self.RFM_connected = False
            if(self.connection == False):
                self.client.connect(self.host, 1883)
                self.client.loop_start()
            time.sleep(0.5)

    def send_file(self, topic, filename):
        filesize = os.path.getsize(filename)
        if(filesize > 0):
            with open(filename, "rb") as f:
                data = f.read(filesize)
                f.close()
            self.client.publish(topic, data)
        else:
            self.error = "no such file or directory"
            self.new_error = True

    def PA_SEND(self):
        self.send_file(self.npnt_pa_topic, self.PA_filename)

    def on_closing(self):
        self.client.loop_stop()