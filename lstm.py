import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os

sub_dirs = ['boxing', 'jack', 'jump', 'squats']


def one_hot_encoding(y_data, sub_dirs):
    Mapping = dict()
    count = 0
    for i in sub_dirs:
        Mapping[i] = count
        count = count+1

    y_features2 = []
    for i in range(len(y_data)):
        Type = str(y_data[i][0])  # 生成体素时写错了一点，y_data[i]才是正确的，但懒得改了
        lab = Mapping[Type]
        y_features2.append(lab)
    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features, num_classes=len(sub_dirs))
    return np.array(y_features)


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


def get_files(data_dir):
    file = []
    for sub_dir in os.listdir(data_dir):
        files = os.listdir(os.path.join(data_dir, sub_dir))

        file.extend([(sub_dir, file) for file in files])
    return file


def voxel_data_generator(data_dir, file_list, batch_size=1, is_validation=False):
    while True:
        np.random.shuffle(file_list)

        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i + batch_size]
            all_data = []

            for sub_dir, file_name in batch_files:
                file_path = os.path.join(data_dir, sub_dir, file_name)
                data = np.load(file_path)
                X_data, y_data = data['arr_0'], data['arr_1']

                # 根据 is_validation 判断是否是验证集
                if is_validation:
                    a, X_data = train_test_split(
                        X_data, test_size=0.2, random_state=1)
                    a, y_data = train_test_split(
                        y_data, test_size=0.2, random_state=1)
                else:
                    X_data, a = train_test_split(
                        X_data, test_size=0.2, random_state=1)
                    y_data, a = train_test_split(
                        y_data, test_size=0.2, random_state=1)
                X_data = X_data.reshape(
                    X_data.shape[0], X_data.shape[1], X_data.shape[2]*X_data.shape[3]*X_data.shape[4])
                y_data = one_hot_encoding(y_data, sub_dirs)
                all_data.append((X_data, y_data))

            batch_data = np.concatenate([data[0] for data in all_data])
            batch_labels = np.concatenate([data[1] for data in all_data])
            yield np.array(batch_data), np.array(batch_labels)


# 定义数据路径和模型保存路径
extract_path = r'E:\RadHAR\voxels\Train'
checkpoint_model_path = r"E:\RadHAR-master\lstm\lstm5.h5"

# 其他参数
batch_size = 1
epochs = 30

# 初始化模型
model = full_3D_model()

# 编译模型
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# 划分训练集和验证集
dirs = get_files(extract_path)

# 计算每个 epoch 的总训练和验证步数
total_train_steps = len(dirs) // batch_size
total_val_steps = len(dirs) // batch_size

# 创建训练集和验证集的生成器
train_generator = voxel_data_generator(
    extract_path, file_list=dirs, batch_size=batch_size)
val_generator = voxel_data_generator(
    extract_path, file_list=dirs, batch_size=batch_size, is_validation=True)

# ModelCheckpoint 回调
checkpoint_model_path = r"E:\RadHAR\lstm2\lstm_model_epoch_{epoch:02d}.h5"
checkpoint = ModelCheckpoint(
    checkpoint_model_path, monitor='val_loss', verbose=1, save_best_only=False, save_freq='epoch')
callbacks_list = [checkpoint]


# Training the model using generators
learning_hist = model.fit(train_generator,
                          steps_per_epoch=total_train_steps,
                          epochs=epochs,
                          validation_data=val_generator,
                          validation_steps=total_val_steps,
                          callbacks=callbacks_list)
for epoch, acc, val_acc in zip(range(1, epochs + 1), learning_hist.history['accuracy'], learning_hist.history['val_accuracy']):
    print(
        f'Epoch {epoch}: Training Accuracy - {acc}, Validation Accuracy - {val_acc}')
