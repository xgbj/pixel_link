# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

import util



def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))


# def convert_to_example(image_data, filename, labels, labels_text, bboxes, oriented_bboxes, shape):
#     """Build an Example proto for an image example.
#     Args:
#       image_data: string, JPEG encoding of RGB image;
#       labels: list of integers, identifier for the ground truth;
#       labels_text: list of strings, human-readable labels;
#       oriented_bboxes: list of bounding oriented boxes; each box is a list of floats in [0, 1];
#           specifying [x1, y1, x2, y2, x3, y3, x4, y4]
#       bboxes: list of bbox in rectangle, [xmin, ymin, xmax, ymax] 
#     Returns:
#       Example proto
#     """
#     
#     image_format = b'JPEG'
#     oriented_bboxes = np.asarray(oriented_bboxes)
#     bboxes = np.asarray(bboxes)
#     example = tf.train.Example(features=tf.train.Features(feature={
#             'image/shape': int64_feature(list(shape)),
#             'image/object/bbox/xmin': float_feature(list(bboxes[:, 0])),
#             'image/object/bbox/ymin': float_feature(list(bboxes[:, 1])),
#             'image/object/bbox/xmax': float_feature(list(bboxes[:, 2])),
#             'image/object/bbox/ymax': float_feature(list(bboxes[:, 3])),
#             'image/object/bbox/x1': float_feature(list(oriented_bboxes[:, 0])),
#             'image/object/bbox/y1': float_feature(list(oriented_bboxes[:, 1])),
#             'image/object/bbox/x2': float_feature(list(oriented_bboxes[:, 2])),
#             'image/object/bbox/y2': float_feature(list(oriented_bboxes[:, 3])),
#             'image/object/bbox/x3': float_feature(list(oriented_bboxes[:, 4])),
#             'image/object/bbox/y3': float_feature(list(oriented_bboxes[:, 5])),
#             'image/object/bbox/x4': float_feature(list(oriented_bboxes[:, 6])),
#             'image/object/bbox/y4': float_feature(list(oriented_bboxes[:, 7])),
#             'image/object/bbox/label': int64_feature(labels),
#             'image/object/bbox/label_text': bytes_feature(labels_text),
#             'image/format': bytes_feature(image_format),
#             'image/filename': bytes_feature(filename),
#             'image/encoded': bytes_feature(image_data)}))
#     return example

def convert_to_example(image_data, filename, labels, labels_text, bboxes, oriented_bboxes, shape):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      oriented_bboxes: list of bounding oriented boxes; each box is a list of floats in [0, 1];
          specifying [x1, y1, x2, y2, x3, y3, x4, y4]
      bboxes: list of bbox in rectangle, [xmin, ymin, xmax, ymax] 
    Returns:
      Example proto
    """
     
    image_format = b'JPEG'
    oriented_bboxes = np.asarray(oriented_bboxes)
    if len(bboxes) == 0:
        print filename, 'has no bboxes'
     
    bboxes = np.asarray(bboxes)
    def get_list(obj, idx):
        if len(obj) > 0:
            return list(obj[:, idx])
        return []
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/shape': int64_feature(list(shape)),
            'image/object/bbox/xmin': float_feature(get_list(bboxes, 0)),
            'image/object/bbox/ymin': float_feature(get_list(bboxes, 1)),
            'image/object/bbox/xmax': float_feature(get_list(bboxes, 2)),
            'image/object/bbox/ymax': float_feature(get_list(bboxes, 3)),
            'image/object/bbox/x1': float_feature(get_list(oriented_bboxes, 0)),
            'image/object/bbox/y1': float_feature(get_list(oriented_bboxes, 1)),
            'image/object/bbox/x2': float_feature(get_list(oriented_bboxes, 2)),
            'image/object/bbox/y2': float_feature(get_list(oriented_bboxes, 3)),
            'image/object/bbox/x3': float_feature(get_list(oriented_bboxes, 4)),
            'image/object/bbox/y3': float_feature(get_list(oriented_bboxes, 5)),
            'image/object/bbox/x4': float_feature(get_list(oriented_bboxes, 6)),
            'image/object/bbox/y4': float_feature(get_list(oriented_bboxes, 7)),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(filename),
            'image/encoded': bytes_feature(image_data)}))
    return example

def get_split(split_name, dataset_dir, file_pattern, num_samples, reader=None):
    dataset_dir = util.io.get_absolute_path(dataset_dir)
    
    if util.str.contains(file_pattern, '%'):
        file_pattern = util.io.join_path(dataset_dir, file_pattern % split_name)
    else:
        file_pattern = util.io.join_path(dataset_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/oriented_bbox/x1': slim.tfexample_decoder.Tensor('image/object/bbox/x1'),
        'object/oriented_bbox/x2': slim.tfexample_decoder.Tensor('image/object/bbox/x2'),
        'object/oriented_bbox/x3': slim.tfexample_decoder.Tensor('image/object/bbox/x3'),
        'object/oriented_bbox/x4': slim.tfexample_decoder.Tensor('image/object/bbox/x4'),
        'object/oriented_bbox/y1': slim.tfexample_decoder.Tensor('image/object/bbox/y1'),
        'object/oriented_bbox/y2': slim.tfexample_decoder.Tensor('image/object/bbox/y2'),
        'object/oriented_bbox/y3': slim.tfexample_decoder.Tensor('image/object/bbox/y3'),
        'object/oriented_bbox/y4': slim.tfexample_decoder.Tensor('image/object/bbox/y4'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {0:'background', 1:'text'}
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=items_to_descriptions,
            num_classes=2,
            labels_to_names=labels_to_names)


def convert_to_example_ctw(image_data, filename, labels, labels_text, bboxes, polygon, shape):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      oriented_bboxes: list of bounding oriented boxes; each box is a list of floats in [0, 1];
          specifying [x1, y1, x2, y2, x3, y3, x4, y4]
      bboxes: list of bbox in rectangle, [xmin, ymin, xmax, ymax]
    Returns:
      Example proto
    """

    image_format = b'JPEG'
    polygon = np.asarray(polygon)
    if len(bboxes) == 0:
        print filename, 'has no bboxes'

    bboxes = np.asarray(bboxes)

    def get_list(obj, idx):
        if len(obj) > 0:
            return list(obj[:, idx])
        return []

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/shape': int64_feature(list(shape)),
        'image/object/polygon/xmin': float_feature(get_list(bboxes, 0)),
        'image/object/polygon/ymin': float_feature(get_list(bboxes, 1)),
        'image/object/polygon/xmax': float_feature(get_list(bboxes, 2)),
        'image/object/polygon/ymax': float_feature(get_list(bboxes, 3)),
        'image/object/polygon/x1': float_feature(get_list(polygon, 0)),
        'image/object/polygon/y1': float_feature(get_list(polygon, 1)),
        'image/object/polygon/x2': float_feature(get_list(polygon, 2)),
        'image/object/polygon/y2': float_feature(get_list(polygon, 3)),
        'image/object/polygon/x3': float_feature(get_list(polygon, 4)),
        'image/object/polygon/y3': float_feature(get_list(polygon, 5)),
        'image/object/polygon/x4': float_feature(get_list(polygon, 6)),
        'image/object/polygon/y4': float_feature(get_list(polygon, 7)),
        'image/object/polygon/x5': float_feature(get_list(polygon, 8)),
        'image/object/polygon/y5': float_feature(get_list(polygon, 9)),
        'image/object/polygon/x6': float_feature(get_list(polygon, 10)),
        'image/object/polygon/y6': float_feature(get_list(polygon, 11)),
        'image/object/polygon/x7': float_feature(get_list(polygon, 12)),
        'image/object/polygon/y7': float_feature(get_list(polygon, 13)),
        'image/object/polygon/x8': float_feature(get_list(polygon, 14)),
        'image/object/polygon/y8': float_feature(get_list(polygon, 15)),
        'image/object/polygon/x9': float_feature(get_list(polygon, 16)),
        'image/object/polygon/y9': float_feature(get_list(polygon, 17)),
        'image/object/polygon/x10': float_feature(get_list(polygon, 18)),
        'image/object/polygon/y10': float_feature(get_list(polygon, 19)),
        'image/object/polygon/x11': float_feature(get_list(polygon, 20)),
        'image/object/polygon/y11': float_feature(get_list(polygon, 21)),
        'image/object/polygon/x12': float_feature(get_list(polygon, 22)),
        'image/object/polygon/y12': float_feature(get_list(polygon, 23)),
        'image/object/polygon/x13': float_feature(get_list(polygon, 24)),
        'image/object/polygon/y13': float_feature(get_list(polygon, 25)),
        'image/object/polygon/x14': float_feature(get_list(polygon, 26)),
        'image/object/polygon/y14': float_feature(get_list(polygon, 27)),
        'image/object/polygon/label': int64_feature(labels),
        'image/object/polygon/label_text': bytes_feature(labels_text),
        'image/format': bytes_feature(image_format),
        'image/filename': bytes_feature(filename),
        'image/encoded': bytes_feature(image_data)}))
    return example

def get_split_ctw(split_name, dataset_dir, file_pattern, num_samples, reader=None):
    dataset_dir = util.io.get_absolute_path(dataset_dir)

    if util.str.contains(file_pattern, '%'):
        file_pattern = util.io.join_path(dataset_dir, file_pattern % split_name)
    else:
        file_pattern = util.io.join_path(dataset_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/polygon/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x5': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x6': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x7': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x8': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x9': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x10': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x11': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x12': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x13': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/x14': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y5': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y6': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y7': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y8': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y9': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y10': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y11': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y12': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y13': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/y14': tf.VarLenFeature(dtype=tf.float32),
        'image/object/polygon/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/polygon/'),
        'object/polygon/x1': slim.tfexample_decoder.Tensor('image/object/polygon/x1'),
        'object/polygon/x2': slim.tfexample_decoder.Tensor('image/object/polygon/x2'),
        'object/polygon/x3': slim.tfexample_decoder.Tensor('image/object/polygon/x3'),
        'object/polygon/x4': slim.tfexample_decoder.Tensor('image/object/polygon/x4'),
        'object/polygon/y1': slim.tfexample_decoder.Tensor('image/object/polygon/y1'),
        'object/polygon/y2': slim.tfexample_decoder.Tensor('image/object/polygon/y2'),
        'object/polygon/y3': slim.tfexample_decoder.Tensor('image/object/polygon/y3'),
        'object/polygon/y4': slim.tfexample_decoder.Tensor('image/object/polygon/y4'),
        'object/polygon/x5': slim.tfexample_decoder.Tensor('image/object/polygon/x5'),
        'object/polygon/x6': slim.tfexample_decoder.Tensor('image/object/polygon/x6'),
        'object/polygon/x7': slim.tfexample_decoder.Tensor('image/object/polygon/x7'),
        'object/polygon/x8': slim.tfexample_decoder.Tensor('image/object/polygon/x8'),
        'object/polygon/y5': slim.tfexample_decoder.Tensor('image/object/polygon/y5'),
        'object/polygon/y6': slim.tfexample_decoder.Tensor('image/object/polygon/y6'),
        'object/polygon/y7': slim.tfexample_decoder.Tensor('image/object/polygon/y7'),
        'object/polygon/y8': slim.tfexample_decoder.Tensor('image/object/polygon/y8'),
        'object/polygon/x9': slim.tfexample_decoder.Tensor('image/object/polygon/x9'),
        'object/polygon/x10': slim.tfexample_decoder.Tensor('image/object/polygon/x10'),
        'object/polygon/x11': slim.tfexample_decoder.Tensor('image/object/polygon/x11'),
        'object/polygon/x12': slim.tfexample_decoder.Tensor('image/object/polygon/x12'),
        'object/polygon/y9': slim.tfexample_decoder.Tensor('image/object/polygon/y9'),
        'object/polygon/y10': slim.tfexample_decoder.Tensor('image/object/polygon/y10'),
        'object/polygon/y11': slim.tfexample_decoder.Tensor('image/object/polygon/y11'),
        'object/polygon/y12': slim.tfexample_decoder.Tensor('image/object/polygon/y12'),
        'object/polygon/y13': slim.tfexample_decoder.Tensor('image/object/polygon/y13'),
        'object/polygon/y14': slim.tfexample_decoder.Tensor('image/object/polygon/y14'),
        'object/polygon/x13': slim.tfexample_decoder.Tensor('image/object/polygon/x13'),
        'object/polygon/x14': slim.tfexample_decoder.Tensor('image/object/polygon/x14'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/polygon/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {0: 'background', 1: 'text'}
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/polygon': 'A list of polygon boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
        num_classes=2,
        labels_to_names=labels_to_names)