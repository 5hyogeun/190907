import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class FashionChecker:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def create_model(self) -> []:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # plt.figure()
        # plt.imshow(train_images[10])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()
        # train_images = train_images / 255.0
        # test_images = test_images / 255.0
        # modeling
        model = keras.Sequential([     # 순차적으로 층을 쌓음
            keras.layers.Flatten(input_shape=(28,28)),    # multi layer => 딥러닝
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')    # 마지막은 거의 softmax => 확률을 잡아줌
        ])
        model.compile(optimizer='adam',
                      loss = 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # Learning
        model.fit(train_images, train_labels, epochs=5)
        # test
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('테스트 정확도 : {}'.format(test_acc))