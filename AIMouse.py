import numpy as np

from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from collections import deque
import pyautogui
import matplotlib.pyplot as plt

def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process
    # ksp.startSetting(SettingConfigs)        # Start the setting process in sub_thread

num = 1
n = 1
model = tf.keras.models.load_model('3d_cnn_model.h5')
ReadMap = np.zeros([32, 64], dtype=np.float16)
ReadMap1 = np.zeros([32, 64], dtype=np.float16)
ReadMap2 = np.zeros([32, 64], dtype=np.float16)
frame_queue = deque(maxlen=5)
def startLoop():
    # kgl.ksoclib.switchLogMode(True)
    # R = RawDataReceiver(chirps=32)

    # Receiver for getting Raw data
    R = FeatureMapReceiver(chirps=32)       # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(45)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)                             # Trigger receiver before getting the data
    time.sleep(0.5)
    global num, ReadMap, ReadMap1, ReadMap2, model, n, frame_queue
    # global num
    print('# ======== Start getting gesture ===========')
    while True:                             # loop for getting the data

        res = R.getResults()
        # Get data from receiver

        scaler = MinMaxScaler()
        if num==1:
            RDI_Map1 = np.array(res[0], dtype=np.float16)
            # RDI_Map1 = RDI_Map1.tolist()
            PHD_Map1 = np.array(res[1], dtype=np.float16)
            # PHD_Map1 = PHD_Map1.tolist()
            ReadMap[0:32, 0:32] = RDI_Map1
            ReadMap[0:32, 32:65] = PHD_Map1
            ReadMap1 = scaler.fit_transform(ReadMap)
            frame_queue.append(ReadMap)
            # ReadMap1 = ReadMap1.tolist()
            num = 2
        else:
            RDI_Map2 = np.array(res[0], dtype=np.float16)
            # RDI_Map2 = RDI_Map2.tolist()
            PHD_Map2 = np.array(res[1], dtype=np.float16)
            # PHD_Map2 = PHD_Map2.tolist()
            ReadMap[0:32, 0:32] = RDI_Map2
            ReadMap[0:32, 32:65] = PHD_Map2
            ReadMap2 = scaler.fit_transform(ReadMap)
            frame_queue.append(ReadMap)
            # ReadMap2 = ReadMap2.tolist()
            num = 1

        Dif = np.abs(np.sum(ReadMap1) - np.sum(ReadMap2))

        time.sleep(0.035)

        if Dif>145 and n>5:
            data = np.zeros([32, 64, 45], dtype=np.float16)
            frames_array = np.array(frame_queue)  # 形状为 (10, 32, 64)
            frames_array = np.transpose(frames_array, (1, 2, 0))  # 调整形状为 (32, 64, 10)
            data[:, :, 0:5] = frames_array
            for i in range(40):
                res = R.getResults()
                RDI_Map = np.array(res[0], dtype=np.float16)
                PHD_Map = np.array(res[1], dtype=np.float16)
                data[0:32, 0:32, i+5] = RDI_Map
                data[0:32, 32:65, i+5] = PHD_Map
                time.sleep(0.035)

            height, width, frames = data.shape
            data_reshaped = data.reshape(-1, 1)
            data_normalized_reshaped = scaler.fit_transform(data_reshaped)
            data_normalized = data_normalized_reshaped.reshape(height, width, frames)
            # 添加批次维度和通道维度
            data_normalized = np.expand_dims(data_normalized, axis=0)  # (1, 32, 64, 75)
            data_normalized = np.expand_dims(data_normalized, axis=-1)  # (1, 32, 64, 75, 1)

            # 进行预测
            y_pred = model.predict(data_normalized)

            print("Model Prediction:", y_pred)

            y_pred_class = np.argmax(y_pred, axis=1)

            if y_pred_class == 0:
                pyautogui.keyDown('win')
                pyautogui.press('tab')
                pyautogui.keyUp('win')
            elif y_pred_class == 1:
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                pyautogui.press('left')
                pyautogui.press('left')
                pyautogui.keyUp('alt')
            elif y_pred_class == 2:
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                pyautogui.keyUp('alt')


            print(f"Predicted Class: {y_pred_class}")
            time.sleep(1)

            # 'PatPat': 0, 'Left': 1, 'Right': 2


        # print(Dif)

        if res is None:
            continue
        # print('data = {}'.format(res))
        # print('data = {}'.format(ReadMap2))          # Print results
        n = n+1
        # time.sleep(0.035)

        '''
        Application for the data.
        '''

def main():
    kgl.setLib()

    # kgl.ksoclib.switchLogMode(True)

    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs

    startLoop()                             # Last you can continue to get the data in the loop

if __name__ == '__main__':
    main()
