import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 32
num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print 'x_train shape:', x_train.shape
print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
conv1_features = HYPERPARAM{"type": "INT", "token": "conv1_num_output", "transform": "X2", "min": 10, "max": 18}
conv1_kernel = HYPERPARAM{"type": "INT", "token": "conv1_kernel_size", "transform": "X1", "min": 3, "max": 4}
conv2_features = HYPERPARAM{"type": "INT", "token": "conv2_num_output", "transform": "X2", "min": 10, "max": 18}
conv2_kernel = HYPERPARAM{"type": "INT", "token": "conv2_kernel_size", "transform": "X1", "min": 3, "max": 5}
conv3_features = HYPERPARAM{"type": "INT", "token": "conv3_num_output", "transform": "X2", "min": 10, "max": 18}
conv3_kernel = HYPERPARAM{"type": "INT", "token": "conv3_kernel_size", "transform": "X1", "min": 3, "max": 5}
conv4_features = HYPERPARAM{"type": "INT", "token": "conv4_num_output", "transform": "X2", "min": 10, "max": 18}
ip1_output = HYPERPARAM{"type": "INT", "token": "ip1_num_output", "transform": "X50", "min": 6, "max": 14}

model.add(Conv2D(conv1_features, (conv1_kernel, conv1_kernel), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(conv2_features, (conv2_kernel, conv2_kernel)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(conv3_features, (conv3_kernel, conv3_kernel), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(conv4_features, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(ip1_output))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

momentum_val = HYPERPARAM{"type":"FLOAT", "token": "momentum", "min": 0.95, "max": 0.999}
decay_val = HYPERPARAM{"type":"INT", "token": "weight_decay_base", "transform": "NEGEXP10", "min": 1, "max": 5}
lr_val = HYPERPARAM{"type":"INT", "token": "base_lr_base", "transform": "NEGEXP10", "min": 1, "max": 5}

# Instead use SGD as Caffe
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=lr_val, decay=decay_val, momentum=momentum_val),
              metrics=['accuracy'])
