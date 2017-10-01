import tensorflow as tf

IMAGE_SIZE = 224
IMAGE_DEPTH = 3


def read_and_decode(tfrecords_file, batch_size, num_class):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])
    # image = tf.transpose(image, (1, 2, 0))  # convert from D/H/W to H/W/D
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    label = tf.cast(img_features['label'], tf.int32)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    # print(image_batch)
    label_batch = tf.one_hot(label_batch, depth=num_class)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, num_class])
    # print(label_batch)
    return image_batch, label_batch
