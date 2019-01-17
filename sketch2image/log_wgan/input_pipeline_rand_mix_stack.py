import os
import cv2
import numpy as np
import tensorflow as tf
from data_processing.tfrecord import *


# photo_dir = './training_data/photo'
# sketch_dir = './training_data/sketch'
paired_dir_1 = './training_data/sketchy_2'
# paired_dir_2 = '/media/cwl/data1/flickr_output_2'
paired_dir_2 = './training_data/flickr_output_new'
# paired_dir = './training_data/sketchy_mix'

# photo_filenames = [os.path.join(photo_dir, f) for f in os.listdir(photo_dir)
#                    if os.path.isfile(os.path.join(photo_dir, f))]
# sketch_filenames = [os.path.join(sketch_dir, f) for f in os.listdir(sketch_dir)
#                     if os.path.isfile(os.path.join(sketch_dir, f))]
paired_filenames_1 = [os.path.join(paired_dir_1, f) for f in os.listdir(paired_dir_1)
                      if os.path.isfile(os.path.join(paired_dir_1, f))]
paired_filenames_2 = [os.path.join(paired_dir_2, f) for f in os.listdir(paired_dir_2)
                      if os.path.isfile(os.path.join(paired_dir_2, f))]

# print("photo file num: %d" % len(photo_filenames))
# print("sketch file num: %d" % len(sketch_filenames))
print("paired file sketchy num: %d" % len(paired_filenames_1))
print("paired file flickr num: %d" % len(paired_filenames_2))

# build class map
class_mapping = []
classes_info = './data_processing/classes.csv'
classes = read_csv(classes_info)
classes_id = [item['Name'] for item in classes]
for name in paired_filenames_1:
    name = os.path.splitext(os.path.split(name)[1])[0].split('_coco_')[0]
    class_id = classes_id.index(name)
    if class_id not in class_mapping:
        class_mapping.append(class_id)
class_mapping = sorted(class_mapping)
for name in paired_filenames_2:
    name = os.path.splitext(os.path.split(name)[1])[0].split('_coco_')[0]
    class_id = classes_id.index(name)
    if class_id not in class_mapping:
        print(name)
        raise RuntimeError
num_classes = len(class_mapping)


def get_num_classes():
    return num_classes


def map_class_id_to_labels(batch_class_id, class_mapping=class_mapping):
    batch_class_id_backup = tf.identity(batch_class_id)
    # assign_ops = []
    # for i in range(len(class_mapping)):
    #     comparison = tf.equal(batch_class_id_backup, tf.constant(class_mapping[i], dtype=tf.int64))
    #     assignment_op = tf.assign(batch_class_id, tf.where(comparison, tf.ones_like(batch_class_id) * i, batch_class_id))
    #     assign_ops.append(assignment_op)
    # assignment_ops = tf.group(assign_ops)
    # with tf.control_dependencies([assignment_ops]):
    #     ret_tensor = tf.squeeze(tf.one_hot(batch_class_id, len(class_mapping),
    #                                        on_value=1, off_value=0, axis=1, dtype=tf.int64))

    for i in range(num_classes):
        comparison = tf.equal(batch_class_id_backup, tf.constant(class_mapping[i], dtype=tf.int64))
        batch_class_id = tf.where(comparison, tf.ones_like(batch_class_id) * i, batch_class_id)
    ret_tensor = tf.squeeze(tf.one_hot(tf.cast(batch_class_id, dtype=tf.int32), num_classes,
                                       on_value=1, off_value=0, axis=1))
    return ret_tensor


SKETCH_CHANNEL = 3
SIZE_1 = {True: [64, 64],
          False: [64, 64]}
SIZE_2 = {True: [64, 64],
          False: [256, 256]}


def get_paired_input_0(paired_filenames, test_mode, distance_map=True, small=True, data_format='NCHW'):
    if test_mode:
        num_epochs = 1
    else:
        num_epochs = None
    filename_queue = tf.train.string_input_producer(
        paired_filenames, capacity=512, shuffle=True, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'ImageNetID': tf.FixedLenFeature([], tf.string),
            'SketchID': tf.FixedLenFeature([], tf.int64),
            'Category': tf.FixedLenFeature([], tf.string),
            'CategoryID': tf.FixedLenFeature([], tf.int64),
            'Difficulty': tf.FixedLenFeature([], tf.int64),
            'Stroke_Count': tf.FixedLenFeature([], tf.int64),
            'WrongPose': tf.FixedLenFeature([], tf.int64),
            'Context': tf.FixedLenFeature([], tf.int64),
            'Ambiguous': tf.FixedLenFeature([], tf.int64),
            'Error': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64),
            'is_test': tf.FixedLenFeature([], tf.int64),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
            'image_small_jpeg': tf.FixedLenFeature([], tf.string),
            'sketch_png': tf.FixedLenFeature([], tf.string),
            'sketch_small_png': tf.FixedLenFeature([], tf.string),
            'dist_map_png': tf.FixedLenFeature([], tf.string),
            'dist_map_small_png': tf.FixedLenFeature([], tf.string),
        }
    )

    if small:
        image_large = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (64, 64, 3))
        # # resize
        # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

        sketch_large = tf.image.decode_png(features['dist_map_small_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_small_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (64, 64, SKETCH_CHANNEL))
        # # resize
        # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    else:
        image_large = tf.image.decode_jpeg(features['image_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (256, 256, 3))
        # # resize
        # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

        sketch_large = tf.image.decode_png(features['dist_map_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (256, 256, SKETCH_CHANNEL))
        # # resize
        # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

    # Augmentation
    # Image
    image_large = tf.image.random_brightness(image_large, max_delta=0.3)
    image_large = tf.image.random_contrast(image_large, lower=0.8, upper=1.2)
    # image_large = tf.image.random_hue(image_large, max_delta=0.05)

    # Fused
    image_pair_fused = tf.concat([image_large, sketch_large], axis=2)
    image_pair_fused = tf.image.random_flip_left_right(image_pair_fused)
    # image_pair_fused = tf.image.random_flip_up_down(image_pair_fused)

    image_augmented = tf.slice(image_pair_fused, [0, 0, 0], [-1, -1, 3])
    sketch_augmented = tf.slice(image_pair_fused, [0, 0, 3], [-1, -1, SKETCH_CHANNEL])

    image_large = image_augmented  # New image from augmented fused tensor
    sketch_large = sketch_augmented  # New sketch from augmented fused tensor

    # Resize
    # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

    image_small = tf.image.resize_images(image_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)
    sketch_small = tf.image.resize_images(sketch_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)

    # Transpose for data format
    if data_format == 'NCHW':
        image_large = tf.transpose(image_large, [2, 0, 1])
        image_small = tf.transpose(image_small, [2, 0, 1])
        sketch_large = tf.transpose(sketch_large, [2, 0, 1])
        sketch_small = tf.transpose(sketch_small, [2, 0, 1])

    # Normalization
    image_large = (image_large - tf.reduce_min(image_large)) / (tf.reduce_max(image_large)
                                                                - tf.reduce_min(image_large) + 1)
    image_large += tf.random_uniform(shape=image_large.shape, minval=0., maxval=1. / 256)  # dequantize
    image_small = (image_small - tf.reduce_min(image_small)) / (tf.reduce_max(image_small)
                                                                - tf.reduce_min(image_small) + 1)
    image_small += tf.random_uniform(shape=image_small.shape, minval=0., maxval=1. / 256)  # dequantize

    # Sketch
    sketch_large = sketch_large / 255.
    sketch_small = sketch_small / 255.

    image_large = image_large * 2. - 1
    image_small = image_small * 2. - 1
    sketch_large = sketch_large * 2. - 1
    sketch_small = sketch_small * 2. - 1

    # Attributes
    class_id = features['class_id']
    is_test = features['is_test']
    # is_test = 0

    Stroke_Count = features['Stroke_Count']
    Difficulty = features['Difficulty']
    CategoryID = features['CategoryID']

    WrongPose = features['WrongPose']
    Context = features['Context']
    Ambiguous = features['Ambiguous']
    Error = features['Error']

    if not test_mode:
        is_valid = WrongPose + Context + Ambiguous + Error + is_test
    else:
        is_valid = 1 - is_test

    is_valid = tf.equal(is_valid, 0)

    return image_small, sketch_small, image_large, sketch_large, class_id, is_valid


def get_paired_input(paired_filenames, test_mode, distance_map=True, small=True, data_format='NCHW'):
    if test_mode:
        num_epochs = 1
    else:
        num_epochs = None
    filename_queue = tf.train.string_input_producer(
        paired_filenames, capacity=512, shuffle=True, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'ImageNetID': tf.FixedLenFeature([], tf.string),
            'SketchID': tf.FixedLenFeature([], tf.int64),
            'Category': tf.FixedLenFeature([], tf.string),
            'CategoryID': tf.FixedLenFeature([], tf.int64),
            'Difficulty': tf.FixedLenFeature([], tf.int64),
            'Stroke_Count': tf.FixedLenFeature([], tf.int64),
            'WrongPose': tf.FixedLenFeature([], tf.int64),
            'Context': tf.FixedLenFeature([], tf.int64),
            'Ambiguous': tf.FixedLenFeature([], tf.int64),
            'Error': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64),
            'is_test': tf.FixedLenFeature([], tf.int64),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
            'image_small_jpeg': tf.FixedLenFeature([], tf.string),
            'sketch_png': tf.FixedLenFeature([], tf.string),
            'sketch_small_png': tf.FixedLenFeature([], tf.string),
            'dist_map_png': tf.FixedLenFeature([], tf.string),
            'dist_map_small_png': tf.FixedLenFeature([], tf.string),
        }
    )

    if small:
        image_large = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (64, 64, 3))
        # # resize
        # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

        sketch_large = tf.image.decode_png(features['dist_map_small_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_small_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (64, 64, SKETCH_CHANNEL))
        # # resize
        # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    else:
        image_large = tf.image.decode_jpeg(features['image_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (256, 256, 3))

        image_small = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=True)
        image_small = tf.cast(image_small, tf.float32)
        image_small = tf.reshape(image_small, (64, 64, 3))

        sketch_large = tf.image.decode_png(features['dist_map_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (256, 256, SKETCH_CHANNEL))

        sketch_small = tf.image.decode_png(features['dist_map_small_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_small_png'], channels=SKETCH_CHANNEL)
        sketch_small = tf.cast(sketch_small, tf.float32)
        sketch_small = tf.reshape(sketch_small, (64, 64, SKETCH_CHANNEL))

    # Augmentation
    # Image
    image_large = tf.image.random_brightness(image_large, max_delta=0.3)
    image_large = tf.image.random_contrast(image_large, lower=0.8, upper=1.2)
    # image_large = tf.image.random_hue(image_large, max_delta=0.05)

    if not small:
        image_small = tf.image.random_brightness(image_small, max_delta=0.3)
        image_small = tf.image.random_contrast(image_small, lower=0.8, upper=1.2)
        # image_small = tf.image.random_hue(image_small, max_delta=0.05)

    # # Fused
    # image_pair_fused = tf.concat([image_large, sketch_large], axis=2)
    # image_pair_fused = tf.image.random_flip_left_right(image_pair_fused)
    # # image_pair_fused = tf.image.random_flip_up_down(image_pair_fused)

    # image_augmented = tf.slice(image_pair_fused, [0, 0, 0], [-1, -1, 3])
    # sketch_augmented = tf.slice(image_pair_fused, [0, 0, 3], [-1, -1, SKETCH_CHANNEL])

    # image_large = image_augmented  # New image from augmented fused tensor
    # sketch_large = sketch_augmented  # New sketch from augmented fused tensor

    # Resize
    # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

    if small:
        image_small = tf.image.resize_images(image_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)
        sketch_small = tf.image.resize_images(sketch_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)

    # Transpose for data format
    if data_format == 'NCHW':
        image_large = tf.transpose(image_large, [2, 0, 1])
        image_small = tf.transpose(image_small, [2, 0, 1])
        sketch_large = tf.transpose(sketch_large, [2, 0, 1])
        sketch_small = tf.transpose(sketch_small, [2, 0, 1])

    # Normalization
    image_large = (image_large - tf.reduce_min(image_large)) / (tf.reduce_max(image_large)
                                                                - tf.reduce_min(image_large) + 1)
    image_large += tf.random_uniform(shape=image_large.shape, minval=0., maxval=1. / 256)  # dequantize
    image_small = (image_small - tf.reduce_min(image_small)) / (tf.reduce_max(image_small)
                                                                - tf.reduce_min(image_small) + 1)
    image_small += tf.random_uniform(shape=image_small.shape, minval=0., maxval=1. / 256)  # dequantize

    # Sketch
    sketch_large = sketch_large / 255.
    sketch_small = sketch_small / 255.

    image_large = image_large * 2. - 1
    image_small = image_small * 2. - 1
    sketch_large = sketch_large * 2. - 1
    sketch_small = sketch_small * 2. - 1

    # Attributes
    class_id = features['class_id']
    is_test = features['is_test']
    # is_test = 0

    Stroke_Count = features['Stroke_Count']
    Difficulty = features['Difficulty']
    CategoryID = features['CategoryID']

    WrongPose = features['WrongPose']
    Context = features['Context']
    Ambiguous = features['Ambiguous']
    Error = features['Error']

    if not test_mode:
        is_valid = WrongPose + Context + Ambiguous + Error + is_test
    else:
        is_valid = 1 - is_test

    is_valid = tf.equal(is_valid, 0)

    return image_small, sketch_small, image_large, sketch_large, class_id, is_valid


def get_paired_input_test(paired_filenames, test_mode=True, distance_map=True, small=True, data_format='NCHW'):
    if test_mode:
        num_epochs = 1
        shuffle = False
    else:
        num_epochs = None
        shuffle = True
    filename_queue = tf.train.string_input_producer(
        paired_filenames, capacity=512, shuffle=shuffle, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'ImageNetID': tf.FixedLenFeature([], tf.string),
            'SketchID': tf.FixedLenFeature([], tf.int64),
            'Category': tf.FixedLenFeature([], tf.string),
            'CategoryID': tf.FixedLenFeature([], tf.int64),
            'Difficulty': tf.FixedLenFeature([], tf.int64),
            'Stroke_Count': tf.FixedLenFeature([], tf.int64),
            'WrongPose': tf.FixedLenFeature([], tf.int64),
            'Context': tf.FixedLenFeature([], tf.int64),
            'Ambiguous': tf.FixedLenFeature([], tf.int64),
            'Error': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64),
            'is_test': tf.FixedLenFeature([], tf.int64),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
            'image_small_jpeg': tf.FixedLenFeature([], tf.string),
            'sketch_png': tf.FixedLenFeature([], tf.string),
            'sketch_small_png': tf.FixedLenFeature([], tf.string),
            'dist_map_png': tf.FixedLenFeature([], tf.string),
            'dist_map_small_png': tf.FixedLenFeature([], tf.string),
        }
    )

    if small:
        image_large = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (64, 64, 3))
        # # resize
        # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

        sketch_large = tf.image.decode_png(features['dist_map_small_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_small_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (64, 64, SKETCH_CHANNEL))
        # # resize
        # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    else:
        image_large = tf.image.decode_jpeg(features['image_jpeg'], fancy_upscaling=True)
        image_large = tf.cast(image_large, tf.float32)
        image_large = tf.reshape(image_large, (256, 256, 3))

        image_small = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=True)
        image_small = tf.cast(image_small, tf.float32)
        image_small = tf.reshape(image_small, (64, 64, 3))

        sketch_large = tf.image.decode_png(features['dist_map_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_png'], channels=SKETCH_CHANNEL)
        sketch_large = tf.cast(sketch_large, tf.float32)
        sketch_large = tf.reshape(sketch_large, (256, 256, SKETCH_CHANNEL))

        sketch_small = tf.image.decode_png(features['dist_map_small_png'], channels=SKETCH_CHANNEL) if distance_map \
            else tf.image.decode_png(features['sketch_small_png'], channels=SKETCH_CHANNEL)
        sketch_small = tf.cast(sketch_small, tf.float32)
        sketch_small = tf.reshape(sketch_small, (64, 64, SKETCH_CHANNEL))

    # Augmentation
    # Image
    image_large = tf.image.random_brightness(image_large, max_delta=0.3)
    image_large = tf.image.random_contrast(image_large, lower=0.8, upper=1.2)
    # image_large = tf.image.random_hue(image_large, max_delta=0.05)

    if not small:
        image_small = tf.image.random_brightness(image_small, max_delta=0.3)
        image_small = tf.image.random_contrast(image_small, lower=0.8, upper=1.2)
        # image_small = tf.image.random_hue(image_small, max_delta=0.05)

    # # Fused
    # image_pair_fused = tf.concat([image_large, sketch_large], axis=2)
    # image_pair_fused = tf.image.random_flip_left_right(image_pair_fused)
    # # image_pair_fused = tf.image.random_flip_up_down(image_pair_fused)

    # image_augmented = tf.slice(image_pair_fused, [0, 0, 0], [-1, -1, 3])
    # sketch_augmented = tf.slice(image_pair_fused, [0, 0, 3], [-1, -1, SKETCH_CHANNEL])

    # image_large = image_augmented  # New image from augmented fused tensor
    # sketch_large = sketch_augmented  # New sketch from augmented fused tensor

    # Resize
    # image_large = tf.image.resize_images(image_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)
    # sketch_large = tf.image.resize_images(sketch_large, size=SIZE_2[small], method=tf.image.ResizeMethod.AREA)

    if small:
        image_small = tf.image.resize_images(image_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)
        sketch_small = tf.image.resize_images(sketch_large, size=SIZE_1[small], method=tf.image.ResizeMethod.AREA)

    # Transpose for data format
    if data_format == 'NCHW':
        image_large = tf.transpose(image_large, [2, 0, 1])
        image_small = tf.transpose(image_small, [2, 0, 1])
        sketch_large = tf.transpose(sketch_large, [2, 0, 1])
        sketch_small = tf.transpose(sketch_small, [2, 0, 1])

    # Normalization
    image_large = (image_large - tf.reduce_min(image_large)) / (tf.reduce_max(image_large)
                                                                - tf.reduce_min(image_large) + 1)
    image_large += tf.random_uniform(shape=image_large.shape, minval=0., maxval=1. / 256)  # dequantize
    image_small = (image_small - tf.reduce_min(image_small)) / (tf.reduce_max(image_small)
                                                                - tf.reduce_min(image_small) + 1)
    image_small += tf.random_uniform(shape=image_small.shape, minval=0., maxval=1. / 256)  # dequantize

    # Sketch
    sketch_large = sketch_large / 255.
    sketch_small = sketch_small / 255.

    image_large = image_large * 2. - 1
    image_small = image_small * 2. - 1
    sketch_large = sketch_large * 2. - 1
    sketch_small = sketch_small * 2. - 1

    # Attributes
    category = features['Category']
    imagenet_id = features['ImageNetID']
    sketch_id = features['SketchID']

    class_id = features['class_id']
    is_test = features['is_test']
    # is_test = 0

    Stroke_Count = features['Stroke_Count']
    Difficulty = features['Difficulty']
    CategoryID = features['CategoryID']

    WrongPose = features['WrongPose']
    Context = features['Context']
    Ambiguous = features['Ambiguous']
    Error = features['Error']

    if not test_mode:
        is_valid = WrongPose + Context + Ambiguous + Error + is_test
    else:
        is_valid = 1 - is_test

    is_valid = tf.equal(is_valid, 0)

    return image_small, sketch_small, image_large, sketch_large, class_id, is_valid, category, imagenet_id, sketch_id


def build_input_queue_paired_sketchy(batch_size, img_dim, test_mode,
                                     data_format='NCHW', distance_map=True, small=True, capacity=8192):
    image_small, sketch_small, image_large, sketch_large, class_id, is_valid = get_paired_input(
        paired_filenames_1, test_mode, distance_map=distance_map, small=small, data_format=data_format)

    images_small, sketches_small, images_large, sketches_large, class_ids = tf.train.maybe_shuffle_batch(
        [image_small, sketch_small, image_large, sketch_large, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=32,
        num_threads=10)

    return images_small, sketches_small, images_large, sketches_large, map_class_id_to_labels(class_ids)


def build_input_queue_paired_sketchy_test(batch_size, img_dim, test_mode=True,
                                          data_format='NCHW', distance_map=True, small=True, capacity=8192):
    image_small, sketch_small, image_large, sketch_large, class_id, is_valid, category, imagenet_id, sketch_id = get_paired_input_test(
        paired_filenames_1, True, distance_map=distance_map, small=small, data_format=data_format)

    images_small, sketches_small, images_large, sketches_large, class_ids, categories, imagenet_ids, sketch_ids = tf.train.maybe_batch(
        [image_small, sketch_small, image_large, sketch_large, class_id, category, imagenet_id, sketch_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, num_threads=2)

    return images_small, sketches_small, images_large, sketches_large, map_class_id_to_labels(class_ids), categories, imagenet_ids, sketch_ids


def build_input_queue_paired_flickr(batch_size, img_dim, test_mode,
                                    data_format='NCHW', distance_map=True, small=True, capacity=int(1.5 * 2 ** 15)):
    image_small, sketch_small, image_large, sketch_large, class_id, is_valid = get_paired_input(
        paired_filenames_2, test_mode, distance_map=distance_map, small=small, data_format=data_format)

    images_small, sketches_small, images_large, sketches_large, class_ids = tf.train.maybe_shuffle_batch(
        [image_small, sketch_small, image_large, sketch_large, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=512,
        num_threads=10)

    return images_small, sketches_small, images_large, sketches_large, map_class_id_to_labels(class_ids)


def build_input_queue_paired_mixed(batch_size, img_dim, test_mode, portion=None,
                                   data_format='NCHW', distance_map=True, small=True, capacity=int(1.5 * 2 ** 15)):
    def _sk_list():
        image_small_sk, sketch_small_sk, image_large_sk, sketch_large_sk, class_id_sk, is_valid_sk = get_paired_input(
            paired_filenames_1, test_mode, distance_map=distance_map, small=small, data_format=data_format)
        return image_small_sk, sketch_small_sk, image_large_sk, sketch_large_sk, class_id_sk, is_valid_sk

    def _f_list():
        image_small_f, sketch_small_f, image_large_f, sketch_large_f, class_id_f, is_valid_f = get_paired_input(
            paired_filenames_2, test_mode, distance_map=distance_map, small=small, data_format=data_format)
        return image_small_f, sketch_small_f, image_large_f, sketch_large_f, class_id_f, is_valid_f

    idx = tf.floor(tf.random_uniform(shape=(), minval=0., maxval=1., dtype=tf.float32) + portion)
    sk_list = _sk_list()
    f_list = _f_list()
    image_small, sketch_small, image_large, sketch_large, class_id, is_valid = [
        tf.cast(a, tf.float32) * idx + tf.cast(b, tf.float32) * (1 - idx) for a, b in zip(sk_list, f_list)
    ]
    class_id = tf.cast(class_id, tf.int64)
    is_valid = tf.cast(is_valid, tf.bool)
    # is_valid = tf.Print(is_valid, [idx, sk_list[4], f_list[4], class_id, sk_list[5], f_list[5], is_valid])

    images_small, sketches_small, images_large, sketches_large, class_ids = tf.train.maybe_shuffle_batch(
        [image_small, sketch_small, image_large, sketch_large, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=512,
        num_threads=10)

    return images_small, sketches_small, images_large, sketches_large, map_class_id_to_labels(class_ids)


def split_inputs(input_data, batch_size, batch_portion, num_gpu):
    input_data_list = []
    dim = len(input_data.get_shape())
    start = 0
    for i in range(num_gpu):
        idx = [start]
        size = [batch_size * batch_portion[i]]
        idx.extend([0] * (dim - 1))
        size.extend([-1] * (dim - 1))
        input_data_list.append(tf.slice(input_data, idx, size))

        start += batch_size * batch_portion[i]
    return input_data_list
