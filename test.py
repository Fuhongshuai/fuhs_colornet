import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import train2
import glob


current_dir = os.getcwd()
save_path = current_dir + '/checkpoints/'
train_dir = current_dir + '/test/'
#print(train_dir)
test_dir = current_dir + '/test1/'
filenames = sorted(glob.glob(train_dir + "/*.png"))
#print(filenames)

phase_train = tf.placeholder(tf.bool, name='phase_train')
uv = tf.placeholder(tf.uint8, name='uv')

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
                                    [0.587, -0.331, -0.418],
                                    [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant([[[[1., 1., 1.],
                                    [0., -0.34413999, 1.77199996],
                                    [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.multiply(tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp

def read_my_file_format(filename_queue, randomize=False):
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=3)
    uint8image = tf.random_crop(uint8image, (224, 224, 3))
    if randomize:
        uint8image = tf.image.random_flip_left_right(uint8image)
        uint8image = tf.image.random_flip_up_down(uint8image, seed=None)
    float_image = tf.div(tf.cast(uint8image, tf.float32), 255)
    return float_image


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example = read_my_file_format(filename_queue, randomize=False)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch([example], batch_size=batch_size, capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)
    return example_batch

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def test():
    #print(img)
    batch_size = 1
    num_epochs = 1
    pred = train2.color_net()
    print(pred)
    colorimage = input_pipeline(filenames, batch_size, num_epochs=num_epochs)

    grayscale = tf.image.rgb_to_grayscale(colorimage)
    grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
    grayscale_yuv = rgb2yuv(grayscale_rgb)
    grayscale = tf.concat([grayscale, grayscale, grayscale], 3)

    pred_yuv = tf.concat([tf.split(grayscale_yuv, 3, 3)[0], pred], 3)
    pred_rgb = yuv2rgb(pred_yuv)
    colorimage_yuv = rgb2yuv(colorimage)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess,save_path+'model_epoch31550000.ckpt')
        num = 1
        try:
            while not coord.should_stop():
                pred_rgb_,grayscale_rgb_ = sess.run([pred_rgb,grayscale_rgb],feed_dict={phase_train: False, uv: 3})
                summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
                #summary_image = concat_images(summary_image, pred_rgb_[0])
                plt.imsave(test_dir+str(num)+'shuai.jpg', summary_image)
                num += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':

    test()
