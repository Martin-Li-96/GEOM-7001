#keras                        3.4.1
#tensorflow                   2.17.0
#tensorflow-io-gcs-filesystem 0.37.1

import tensorflow as tf
from keras.src.legacy.backend import l2_normalize
from pygments.lexer import include
from tensorflow.keras import regularizers
import numpy as np
import time
import pandas as pd
from datetime import datetime



with np.load("./Dataset_whole_90_10.npz") as data:
    train_x=data["train_x"][:,:,:,[3,2,1]]
    train_labels=data['train_y']
    test_x=data["test_x"][:,:,:,[3,2,1]]
    test_labels=data['test_y']








class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy >= 0.9:
            print(f"\nReached 90% validation accuracy, stopping training at epoch {epoch+1}")
            self.model.stop_training = True





checkpoint_filepath = './RGB_RESNET/RGB_RESNET-{epoch:02d}-{val_accuracy:.4f}.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

logdir="resnet_log/fit/" + "Fold3-1"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_labels))


test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_labels))

# 954*4
BATCH_SIZE =954*3
SHUFFLE_BUFFER_SIZE = len(train_x)

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"],
                                          cross_device_ops=tf.distribute.NcclAllReduce())

start_time = time.time()


with strategy.scope():


    # input_layer=tf.keras.Input(shape=(64,64,13))
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet50V2(weights='imagenet',include_top=False,input_shape=(64,64,3),pooling='max'),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

history=model.fit(train_dataset, epochs=200,verbose=1,validation_data=test_dataset,callbacks=[CustomCallback(),model_checkpoint_callback, tensorboard_callback ])

end_time = time.time()
training_time = end_time - start_time

print(f"Training took {training_time/60:.2f} mins")
