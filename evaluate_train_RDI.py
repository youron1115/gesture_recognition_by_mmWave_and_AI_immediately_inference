import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models

import wandb

current_path = os.path.dirname(os.path.abspath(__file__))

dense_hidden_units = 256
dropout_rate = 0.3
#epochs = 
LSTM_units = 64
test_st="1st_label_is_middle"

wandb.init(
    project=f'RDI_gesture_model_slide_window_test_grid_{test_st}',
    name=f'RDI_gesture_model_{test_st}_{dense_hidden_units}_{dropout_rate}',
    config={
        'batch_size': 180,
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
            
    }
)

config= wandb.config

def evaluate_model():   
    
    processed_data_path=os.path.join(current_path, "processed_data")
    train_data = np.load(os.path.join(processed_data_path, 'train.npz'))
    train_labels = train_data['labels']
    train_data = train_data['data']
    #train_data = train_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("train_data shape:", train_data.shape)

    
    valid_data = np.load(os.path.join(processed_data_path, 'val.npz'))
    valid_labels = valid_data['labels']
    valid_data = valid_data['data']
    #valid_data = valid_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("valid_data shape:", valid_data.shape)
    
    test_data = np.load(os.path.join(processed_data_path, 'test.npz'))
    test_labels = test_data['labels']
    test_data = test_data['data']
    #test_data = test_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("test_data shape:", test_data.shape)
    
    wandb.run.name = f"RDI_gesture_model_slide_window_grid_evaluate_{dense_hidden_units}_{LSTM_units}_{dropout_rate}"

    model_dir =os.path.join(current_path, "model")
    
    """
    #資料預測結果(機率)
    model_name=r"D:\gesture_recognition_by_mmWave_and_AI\online_infer\model\RDI_gesture_model_slide_window_32_0.3_64_ep_60.h5"
    model=tf.keras.models.load_model(model_name)
    number=1
    
    for tr in range(len(train_data)):
        print("data No.: ", number)
        number+=1
        load_and_predict(model, train_data[tr])
        print("label: ", train_labels[tr])
        print("========================================\n")
    """
    
    model_name = f"RDI_gesture_model_slide_window_{dense_hidden_units}_{dropout_rate}_{LSTM_units}"
    
    for e in range(10,150+1,10):
        
        path = os.path.join(model_dir, f"{model_name}_ep_{e}.h5")
        loaded_model = tf.keras.models.load_model(path)
        print(f"\nevaluate model : {path}")
        
        wandb.log({
            "epochs": e,
            "train_acc": loaded_model.evaluate(train_data, train_labels)[1],
            "train_loss": loaded_model.evaluate(train_data, train_labels)[0],
            "val_acc": loaded_model.evaluate(valid_data, valid_labels)[1],
            "val_loss": loaded_model.evaluate(valid_data, valid_labels)[0],
            "test_acc": loaded_model.evaluate(test_data, test_labels)[1],
            "test_loss": loaded_model.evaluate(test_data, test_labels)[0],
        })
    
def load_and_predict(model, data):
    print("data shape: ", data.shape)  # (32, 32)
    data=np.expand_dims(data, axis=0)  # Expand dimensions to match model input shape (1, 32, 32)
    data=np.expand_dims(data, axis=-1) # Expand dimensions to match model input shape (1, 32, 32, 1)
    print("model inferring")
    predictions = model.predict(data)
    probs= predictions[0]*100
    probs_str = [f"{p:.3f}%" for p in probs]
    print("predicted probabilities: ", probs_str)
    
        
        
    
evaluate_model()