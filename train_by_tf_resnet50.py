import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np

data_dir = {'train': './train', 'valid': './test'}
img_width = 200
img_height = 200
img_channel = 3
batch_size = 32
epochs = 200
nb_train_samples = 2536
nb_validation_samples = 2636

lable2cat = {key: value for value, key in enumerate(['T62', 'D7', 'ZIL131', '2S1', 'SN_C71', 'ZSU_23_4', 'BRDM_2', 'SN_132', 'BTR_60', 'SN_9563'])}
# print(lable2cat)

train_path_list = []
train_label_list = []
test_path_list = []
test_label_list = []

for phase in data_dir.keys():
    dir = data_dir[phase]
    son_dirs = os.listdir(dir)
    for son_dir in son_dirs:
        data_path = os.listdir(os.path.join(dir, son_dir))
        if phase == 'train':
            train_path_list.extend([os.path.join(dir, son_dir, __) for __ in data_path])
            train_label_list.extend([lable2cat[son_dir]]*len(data_path))
        else:
            test_path_list.extend([os.path.join(dir, son_dir, __) for __ in data_path])
            test_label_list.extend([lable2cat[son_dir]] * len(data_path))
# 训练集
train_x = []
train_y = train_label_list
for __ in train_path_list:
    img = cv2.imread(__)
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    train_x.append(img)

train_x = np.array(train_x)
train_y = np.array(train_y)
# 打乱顺序
shuffle_ix = np.random.permutation(np.arange(len(train_y)))
train_x = train_x[shuffle_ix]
train_y = train_y[shuffle_ix]

# 测试集
test_x = []
test_y = test_label_list
for __ in test_path_list:
    img = cv2.imread(__)
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    test_x.append(img)

test_x = np.array(test_x)
test_y = np.array(test_y)
# 打乱顺序
shuffle_ix = np.random.permutation(np.arange(len(test_y)))
test_x = test_x[shuffle_ix]
test_y = test_y[shuffle_ix]


image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255.,
    shear_range=0.2,
    rotation_range=10.,
    zoom_range=0.2,
    horizontal_flip=True
)
image_gen_train.fit(train_x)

image_gen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255.
)
image_gen_test.fit(test_x)

# train_generator
# train_generator = image_gen_train.flow_from_directory(data_dir['train'], target_size=(img_width, img_height),
#                                                       batch_size=batch_size, class_mode='categorical', shuffle=True)
# test_generator = image_gen_test.flow_from_directory(data_dir['valid'], target_size=(img_width, img_height),
#                                                       batch_size=batch_size, class_mode='categorical', shuffle=True)


# 加载预训练的 resnet50 model
resnet50_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(img_width, img_height, img_channel))
# resnet50_model.summary()

# 重组网络结构
x = resnet50_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(10, 'softmax')(x)
model = tf.keras.Model(inputs=resnet50_model.input, outputs=x)

# 设置 trainable，仅训练后几层卷积 block
# set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
for layer in resnet50_model.layers[:45]:
    layer.trainable = False


# 开始配置 model
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.summary()


# 存储历史模型、加载历史模型
model_save_path = './transfer_check_point/resnet50_transfer_model.ckpt'
if os.path.exists(model_save_path+'.index'):
    print('--------------------load the model---------------------')
    model.load_weights(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True)

# 新增设置tensorboard观察训练过程中的参数变化情况
tensorboard_log_path = './logs/tf'
# 如果该地址不存在就创建一个logs文件夹
if not os.path.exists(tensorboard_log_path):
    os.mkdir(tensorboard_log_path)
# 可以每个batch保存一次loss或者别的信息，指定 updata_freq='batch'即可
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=1)


# model.fit_generator(generator=train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs,
#                     validation_data=test_generator, validation_steps=nb_validation_samples // batch_size,
#                     callbacks=[cp_callback, tensorboard_callback])
model.fit(image_gen_train.flow(x=train_x, y=train_y, batch_size=batch_size), epochs=epochs,
          validation_data=image_gen_test.flow(x=test_x, y=test_y, batch_size=batch_size), validation_freq=1,
          callbacks=[cp_callback, tensorboard_callback])