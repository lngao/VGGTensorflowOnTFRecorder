from datetime import datetime
import math
import time
import VGG as vgg
import numpy as np
import tensorflow as tf

Num_Class = 5
FLAGS = tf.app.flags.FLAGS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'logs/train/',
                           """Directory where to read model checkpoints.""")

def img_read(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal('File does not exists %s', filename)
        # 读取图片数据并且以字符串形式返回
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    # image = tf.ones(shape=[24,24,3],name='input')

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.reshape(image, [1, 32, 32, 3])

    # print(image_data)
    # 将字符串转换成float
    return image

def Inforace(image):
    logits = vgg.VGG16N(image,Num_Class,False)
    pre = tf.arg_max(logits,dimension=1,name='output')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        print(logits.eval())
        print(pre.eval())
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                    output_node_names=['output'])
        with tf.gfile.FastGFile('D:/python_code/04 VGG Tensorflow/logs/classifier.pb',
                            mode='wb') as f:
            f.write(output_graph_def.SerializeToString())



def main(argv=None):
    filename = 'images/1(4).jpg'
    img = img_read(filename)
    Inforace(img)
    #inference(img)


if __name__ == '__main__':
    tf.app.run()
