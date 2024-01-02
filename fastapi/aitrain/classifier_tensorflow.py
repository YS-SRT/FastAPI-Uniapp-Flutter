import tensorflow as tf
import random, math
from PIL import Image
import matplotlib.pyplot as plt, os, pandas as pd, numpy as np
# output chinese word
import matplotlib
matplotlib.rc("font",family='SimHei')

from aitrain.utils import DataPathEnum
from aitrain.dataset.hand_writting_data import HWData

class TFDataset(HWData, tf.keras.utils.Sequence):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.image_files)/self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.image_files))
        
        image_list = []
        label_list = []
        
        for index in range(low, high):
           label, image_path = self.get_image_path_vs_lable(index)
           image_array = np.array(list(Image.open(image_path).getdata()))
           image_list.append(image_array)

           target =[0 if label != i else 1 for i in range(15)]  # one-hot encoding

           label_list.append(target)

        return np.array(image_list).reshape(-1,64,64,1), np.array(label_list)
    
    def on_epoch_end(self):
        random.shuffle(self.image_files)

class TFClassifier():

    def __init__(self, data: TFDataset) -> None:
        model = tf.keras.Sequential()
        layers = tf.keras.layers
        
        model.add(layers.Rescaling(1./255))
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                            input_shape=(64, 64, 1)))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=1024, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dense(units=15, activation='softmax'))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
        self.data = data
        self.checkpoint_file = os.path.join(str(DataPathEnum.MODEL_CHECKPOINT_DIR), "zh_hw_tf.h5")

    def train(self, epochs):
        his = self.model.fit(self.data, epochs=epochs, verbose=2).history
        self.__plot_history(his)
        self.model.summary()

    def random_eval_model(self):
        
        index = random.randint(0, len(self.data.image_files)-1)
        self.data.plot_image(index)

        label, image_path= self.data.get_image_path_vs_lable(index)
        image_array = np.array(list(Image.open(image_path).getdata())).reshape(-1, 64, 64, 1)
        prediction = self.model.predict(image_array)
        df = pd.DataFrame(prediction[0])
        df.plot.barh(rot=0, legend=False, ylim=(0, 15), xlim=(0,1))
        
    def save_model_weights(self):
        self.model.save_weights(self.checkpoint_file)


    def load_model_weights(self):
        self.model.load_weights(self.checkpoint_file)   


    def __plot_history(self, history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(history['loss'], label='training loss')
        # axs[0].plot(history['val_loss'], label='validation loss')
        axs[0].legend(loc='upper left')
        axs[0].set_title('training data vs validation data')

        axs[1].plot(history['accuracy'], label='testing accuracy')
        # axs[1].plot(history['val_accuracy'], label='validation accuracy')
        axs[1].set_ylim([0, 1])
        axs[1].legend(loc='upper left')
        axs[1].set_title('accuracy')

        axs.flat[0].set(xlabel='epochs', ylabel='loss')
        axs.flat[1].set(xlabel='epochs', ylabel='accuracy')

        plt.show()


