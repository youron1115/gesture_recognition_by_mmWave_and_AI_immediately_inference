from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver

import time
import os

import tkinter as tk
from tkinter import ttk
import keyboard

import numpy as np

from collections import deque

import tensorflow as tf
from keras import models

GESTURES = ['background', 'pipi', 'left']

def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_120cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process
    # ksp.startSetting(SettingConfigs)        # Start the setting process in sub_thread

def startLoop(model):
    # kgl.ksoclib.switchLogMode(True)
    #R = RawDataReceiver(chirps=32)

    # Receiver for getting Raw data
    R = FeatureMapReceiver(chirps=32)       # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(100)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)                             # Trigger receiver before getting the data
    #time.sleep(0.5)
    
    print('Press "n" to start')
    keyboard.wait('n') 
    # Wait for the key 'n' to start getting data
    #print('n is pressed')
    
    print('\n# ======== Start getting gesture ===========')
    
    buffer=100#緩衝區
    frame_buffer = deque(maxlen=buffer)
    
    predict_dict = {
        0: 'background',
        1: 'pipi',
        2: 'left',
    }
    
    while True:                             # loop for getting the data
        
        res = R.getResults()                # Get data from receiver
        if res is None:
            print("\nNo data received, continue to next iteration\n")
            continue
        #print("type(res) :", type(res))  # <class 'tuple'>
        print("len :",res[0].shape)         # (32, 32)
        print('data = {}\n'.format(res))          # Print results
        #time.sleep(1)
        '''
        Application for the data.
        '''
        current_frame = res[0]  # Get the current frame data
        frame_buffer.append(current_frame)
        
        if len(frame_buffer) == buffer:
            # If the buffer is full, process the data
            print(f"Buffer size is {len(frame_buffer)}, processing data")
            frame = np.array(frame_buffer)  # Convert the deque to a numpy array
        
        
            print("predicting data")
            predicted_value = load_and_predict(model, frame)
            
        """
        if keyboard.is_pressed('q'):  # Press 'q' to exit the loop or after 100 tries
            print("Exit loop")
            break
        """

def load_and_predict(model, data):
    print("data shape: ", data.shape)  # (32, 32)
    data=np.expand_dims(data, axis=0)  # Expand dimensions to match model input shape (1, 32, 32)
    #print("data shape after 1 expand_dims: ", data.shape)  # (1, 32, 32)
    data=np.expand_dims(data, axis=-1) # Expand dimensions to match model input shape (1, 32, 32, 1)
    #print("data shape after 2 expand_dims: ", data.shape)  # (1
    #p
    print("model inferring")
    predictions = model.predict(data)
    print("transform predictions to argmax")
    predictions = np.argmax(predictions, axis=1)[0]  # Get the index of the maximum value along the last axis
    #print("predictions: ", predictions)
    #print("return")
    return predictions

def startLoop_gui(model):
    R = FeatureMapReceiver(chirps=32)
    R.trigger(chirps=32)

    # ==== Tkinter 視覺化設定 ====
    root = tk.Tk()
    root.title("Gesture Recognition")

    label = tk.Label(root, text="Current gesture: None", font=("Arial", 20), bg="#FFCCCC", width=40)
    label.pack(pady=10)

    bars = []
    for gesture in GESTURES:
        frame = tk.Frame(root)
        frame.pack(side=tk.LEFT, padx=10)

        bar = ttk.Progressbar(frame, orient=tk.VERTICAL, length=100, mode='determinate', maximum=1.0)
        bar.pack()
        bars.append(bar)

        lbl = tk.Label(frame, text=gesture, font=("Arial", 20))
        lbl.pack()

        window_size = 20
        frame_buffer = deque(maxlen=window_size)
        
    def update():
        res = R.getResults()
        if res is not None:
            frame = res[0]  # 單一張圖 shape: (32, 32)
            frame_buffer.append(frame)

            if len(frame_buffer) == window_size:
                # 滿 size才推論
                data_seq = np.array(frame_buffer)               # (100, 32, 32)
                data_seq = np.expand_dims(data_seq, axis=-1)    # (100, 32, 32, 1)
                data_seq = np.expand_dims(data_seq, axis=0)     # (1, 100, 32, 32, 1)

                preds = model.predict(data_seq, verbose=0)[0]   # shape: (num_classes,)
                pred_idx = int(np.argmax(preds))
                pred_name = GESTURES[pred_idx]

                # 更新 GUI
                label.config(text=f"Current gesture: {pred_name}")
                for i, bar in enumerate(bars):
                    bar['value'] = preds[i]

        print("Gesture display started (press 'q' to quit terminal)")
        if keyboard.is_pressed('q'):  # Press 'q' to exit the loop or after 100 tries
            print("Exit loop")
            exit()
        # 每 300ms 更新一次
        root.after(10, update)

    print("Press 'n' to start")
    keyboard.wait('n')
    
    update()
    root.mainloop()

def main():
    print("# ======== Start ===========")
    kgl.setLib()

    # kgl.ksoclib.switchLogMode(True)
    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs
    
    print("load model")
        
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_path, "model")
    model_path=os.path.join(model_dir , "RDI_gesture_model_slide_window_128_0.2_64_ep_60.h5")
    
    #model_path=r"D:\gesture_recognition_by_mmWave_and_AI\online_infer\model"
    
    load_model=models.load_model(model_path)
    #print("summary of model: ", load_model.summary())
    
    #startLoop(load_model)     # Last you can continue to get the data in the loop

    startLoop_gui(load_model)  # Start the GUI loop for gesture recognition
    
if __name__ == '__main__':
    main()
