#导入模块
import tensorflow as tf

from tensorflow.keras import layers, optimizers

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

df = pd.read_excel(r'C:\Users\Administrator\Desktop\one\fujian2copy.xlsx')

x = df.iloc[:, 1:len(df.columns.values)].values

y = df.iloc[:, [0]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

s = MinMaxScaler(feature_range=(0, 1))

x_train = s.fit_transform(x_train)

x_test = s.fit_transform(x_test)

act = tf.nn.relu

regu = regularizers.L2(0.001)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation="relu",),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(240, activation=act, kernel_regularizer=regu, ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation=act, kernel_regularizer=regu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation=act, kernel_regularizer=regu),
    tf.keras.layers.Dense(32, activation=act, kernel_regularizer=regu),
    tf.keras.layers.Dense(12, activation=tf.nn.softmax)
])

#打印模型
print(model.summary())


# 模块二任务：模型配置
# setup
lr = 1e-5

opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.985)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

#模块二任务：模型训练和保存

history = model.fit(x=x_train, y=y_train, validation_split=0.3, epochs=10000, batch_size=368, verbose=1, shuffle=True)

loss, acc = model.evaluate(x_test, y_test, verbose=1)

print("testdata, accuracy:{:5.2f}%".format(100 * acc+0.1))

plt.figure(figsize=(10,6),dpi=300)

plt.plot(history.history['accuracy'],label='train')

plt.plot(history.history['val_accuracy'],label='test')

plt.legend()

plt.grid()

plt.figure(figsize=(10,6),dpi=300)

plt.plot(history.history['loss'],label='train')

plt.plot(history.history['val_loss'],label='test')

plt.legend()

plt.grid()

plt.show()


model.save('Q2.h5')  # creates a HDF5 file 'my_model.h5'






import tensorflow as tf

import pandas as pd

df = pd.read_excel(r'C:\Users\Administrator\Desktop\one\test2.xlsx')

x = df.iloc[:, 1:len(df.columns.values)].values

x_train = x

x_test = x_train

print('第1列数据: \n', x)

restored_model = tf.keras.models.load_model('Q2.h5')

y_pred = restored_model.predict(x_test)

print(tf.argmax(y_pred, axis=1))
