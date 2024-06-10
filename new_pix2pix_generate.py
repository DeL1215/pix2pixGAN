import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 定義下載層
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# 定義上傳層
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# 定義生成器模型
def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# 定義生成圖片的函數
import uuid
def generate_images(model, test_input, output_folder=r"C:\Users\User\Desktop\期末報告\輸出圖片"):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(output_folder + f'/{uuid.uuid4()}_result.png')
    plt.show()

    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig(output_folder + f'/{uuid.uuid4()}_prediction.png')
    plt.show()


# 讀取並準備測試圖片
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    if img.size == (512, 256):
        img = img.crop((0, 0, 256, 256))  # 裁剪左邊256x256部分
    elif img.size != (256, 256):
        img = img.resize((256, 256))  # 將圖片縮放成256x256
    img = np.array(img)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    return img

# 載入檢查點
checkpoint_dir = "C:/Users/User/coding/pix2pixGan/training_checkpoints"
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
generator = Generator()
checkpoint = tf.train.Checkpoint(generator=generator)
status = checkpoint.restore(latest_checkpoint).expect_partial()

# 測試圖片路徑
test_image_path = r'C:\Users\User\Desktop\期末報告\原圖片\test1.png'  # 修改為你自己的測試圖片路徑
test_image = load_image(test_image_path)

# 生成圖片
generate_images(generator, test_image)
