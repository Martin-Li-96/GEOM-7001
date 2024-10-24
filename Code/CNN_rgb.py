#keras                        3.4.1
#tensorflow                   2.17.0
#tensorflow-io-gcs-filesystem 0.37.1

import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import time
import pandas as pd

with np.load("./Dataset_whole_90_10.npz") as data:
    train_x=data["train_x"][:,:,:,[3,2,1]]
    train_labels=data['train_y']
    test_x=data["test_x"][:,:,:,[3,2,1]]
    test_labels=data['test_y']

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy >= 0.91:
            print(f"\nReached 90% validation accuracy, stopping training at epoch {epoch+1}")
            self.model.stop_training = True






checkpoint_filepath = './RGB_CNN/RGB_CNN-{epoch:02d}-{val_accuracy:.2f}.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_labels))


test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_labels))

# 954*4
BATCH_SIZE =954*3
SHUFFLE_BUFFER_SIZE = len(train_x)

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"],
                                          cross_device_ops=tf.distribute.NcclAllReduce())
#devices=["/gpu:0"]

start_time = time.time()

with strategy.scope():
    input_layer=tf.keras.Input(shape=(64,64,3))
    model = tf.keras.Sequential([
        input_layer,
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.Conv2D(128,(4,4),activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((4,4)),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu',kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Conv2D(512, (2, 2), activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

history=model.fit(train_dataset, epochs=120,verbose=1,validation_data=test_dataset,callbacks=[CustomCallback(),model_checkpoint_callback])
# callbacks=[CustomCallback()]


end_time = time.time()

print(end_time-start_time)