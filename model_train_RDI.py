import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

device = None
if tf.config.list_physical_devices('GPU'):
    device = tf.config.list_physical_devices('GPU')[0]
else:
    device = tf.config.list_physical_devices('CPU')[0]
print(f"Using device: {device}")

import wandb

sweep_config={
    'name': 'RDI_gesture_model_sweep_slide_window',
    'project': 'RDI_gesture_model_slide_window',
    
    'method': 'grid',
    
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    
    'parameters': {
        'dense_hidden_units': {
            'values': [32, 64, 128,256]
        },
        'dropout_rate': {
            'values': [0.2, 0.3]#, 0.4, 0.5]
        },
        
        'lstm_units': {
            'value': 64#[32,64,128
            #           ]
        },
        'epochs': {
            'value': 150#[100,200,300
            #           ]
        },
    }
}

current_path = os.path.dirname(os.path.abspath(__file__))

class CustomCheckpoint(Callback):
    def __init__(self, save_path_base, name, save_every_n_epoch=10):
        super().__init__()
        self.save_every_n_epoch = save_every_n_epoch
        self.save_path_base = save_path_base
        self.name = name
        os.makedirs(save_path_base, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.save_every_n_epoch == 0:
            save_path = os.path.join(self.save_path_base, f"{self.name}_ep_{epoch+1}.h5")
            self.model.save(save_path)
            print(f"\nSaved model at: {save_path}")

def fit_model():   
    
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
    
    time_steps = 20
    height = 32
    width = 32
    input_shape = (time_steps, width, height, 1)
    num_classes= 3 
    
    wandb.init(
        project='RDI_gesture_model_slide_window',
        config={
            'batch_size': 180,
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            #'CNN_dropout_rate': 0.3,
        }
    )
    
    config= wandb.config
    dense_hidden_units = config.dense_hidden_units
    dropout_rate = config.dropout_rate
    epochs = config.epochs
    LSTM_units = config.lstm_units
    wandb.run.name = f"RDI_gesture_model_slide_window_{dense_hidden_units}_{LSTM_units}_{dropout_rate}_{epochs}"
    
    model = tf.keras.Sequential([
        #timedistributed is to remain the time structure of the input data
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'),input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((3, 3))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Flatten()),
        
        layers.LSTM(LSTM_units),
        
        layers.Dense(dense_hidden_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model_dir =os.path.join(current_path, "model")
    model_name = f"RDI_gesture_model_slide_window_{dense_hidden_units}_{dropout_rate}_{LSTM_units}"#_{epochs}.h5"
    
    custom_checkpoint= CustomCheckpoint(
        save_path_base=model_dir,
        save_every_n_epoch=10,
        name=model_name
    )
    
    model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), 
              epochs=epochs, 
              batch_size=config.batch_size, 
              shuffle=True,
              callbacks=[custom_checkpoint])
    
    print("\nTraining complete")
    """
    for e in range(10,epochs+1,10):
        
        path = os.path.join(model_dir, f"{model_name}_ep_{e}.h5")
        loaded_model = tf.keras.models.load_model(path)
        print(f"evaluate model : {path}")
        
        wandb.log({
            "epochs": e,
            "train_acc": loaded_model.evaluate(train_data, train_labels)[1],
            "train_loss": loaded_model.evaluate(train_data, train_labels)[0],
            "val_acc": loaded_model.evaluate(valid_data, valid_labels)[1],
            "val_loss": loaded_model.evaluate(valid_data, valid_labels)[0],
            "test_acc": loaded_model.evaluate(valid_data, valid_labels)[1],
            "test_loss": loaded_model.evaluate(valid_data, valid_labels)[0],
            })
    """
    
    """
    
    model.save(os.path.join(model_dir, model_name))
    print("\nsave complete")
    """
"""
fit_model()
"""
sweep_id = wandb.sweep(sweep_config, project='RDI_gesture_model_slide_window_grid')
wandb.agent(sweep_id, function=fit_model)

