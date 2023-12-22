import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

sub_dirs = ['boxing', 'jack', 'jump', 'squats']
extract_path = r'E:\HAR\voxels\Test\Test'
checkpoint_model_path = r"E:\HAR\lstm2\lstm_model_epoch_17.h5"


def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping = dict()
    count = 0
    for i in sub_dirs:
        Mapping[i] = count
        count = count+1

    y_features2 = []
    for i in range(len(y_data)):
        Type = y_data[i]
        lab = Mapping[Type]
        y_features2.append(lab)

    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features)
    return y_features


def full_3D_model(summary=False):
    print('building the model ... ')
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=False,
                                 stateful=False, input_shape=(60, 10 * 1024))))
    model.add(Dropout(.5, name='dropout_1'))
    model.add(Dense(128, activation='relu', name='DENSE_1'))
    model.add(Dropout(.5, name='dropout_2'))
    model.add(Dense(4, activation='softmax', name='output'))

    return model


model = full_3D_model()
model.build((None, 60, 10 * 1024))
model.load_weights(checkpoint_model_path)


# 加载测试数据
test_data_list = []
test_label_list = []

np.random.seed(1)
tf.random.set_seed(1)


# loading the test data
for action_type in sub_dirs:
    Data_path = extract_path + action_type
    data = np.load(Data_path + '.npz')
    test_data_list.append(data['arr_0'])
    test_label_list.append(data['arr_1'])
    del data

# 合并测试数据和标签
test_data = np.concatenate(test_data_list, axis=0)
test_label = np.concatenate(test_label_list, axis=0)
del test_data_list
del test_label_list
# 打印测试数据和标签的形状
print("Test Data Shape:", test_data.shape)
print("Test Label Shape:", test_label.shape)

# 对测试标签进行独热编码
test_label = one_hot_encoding(test_label, sub_dirs, categories=5)

# 将测试数据reshape成和训练数据一致的形状
test_data = test_data.reshape(
    test_data.shape[0], test_data.shape[1], test_data.shape[2]*test_data.shape[3]*test_data.shape[4])

# 打印reshape后的测试数据形状
print("Reshaped Test Data Shape:", test_data.shape)

# 在验证集上进行预测
y_pred = model.predict(test_data)
# 将预测转为类别标签
predicted_labels = np.argmax(y_pred, axis=1)
true_labels = np.argmax(test_label, axis=1)

# 计算准确率
accuracy = accuracy_score(true_labels, predicted_labels)
print("Validation Accuracy:", accuracy)
