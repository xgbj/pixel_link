# encoding=utf-8
import numpy as np;
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example, convert_to_example_ctw
import config


def cvt_to_tfrecords(output_path, data_path, gt_path):
	image_names = util.io.ls(data_path, '.jpg')  # [0:10];
	print "%d images found in %s" % (len(image_names), data_path);
	with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
		for idx, image_name in enumerate(image_names):
			poly_bboxes = [];
			bboxes = []
			labels = [];
			labels_text = [];
			path = util.io.join_path(data_path, image_name);
			print "\tconverting image: %d/%d %s" % (idx, len(image_names), image_name);
			image_data = tf.gfile.FastGFile(path, 'r').read()

			image = util.img.imread(path, rgb=True);
			shape = image.shape
			h, w = shape[0:2];
			h *= 1.0;
			w *= 1.0;
			image_name = util.str.split(image_name, '.')[0];
			gt_name = image_name + '.txt';
			gt_filepath = util.io.join_path(gt_path, gt_name);
			lines = util.io.read_lines(gt_filepath);

			for line in lines:
				line = util.str.remove_all(line, '\xef\xbb\xbf')
				gt = util.str.split(line, ',');

				xmin, ymin, xmax, ymax = float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])
				poly_bbox = []
				for i in range(2, 16):
					wi = float(gt[i * 2])
					hi = float(gt[i * 2 + 1])
					poly_bbox.append(xmin + wi)
					poly_bbox.append(ymin + hi)

				poly_bbox = np.asarray(poly_bbox) / ([w, h] * 14)

				#oriented_box = [int(gt[i]) for i in range(8)];
				#oriented_box = np.asarray(oriented_box) / ([w, h] * 4);

				poly_bboxes.append(poly_bbox)
				#oriented_bboxes.append(oriented_box);

				xs = poly_bbox.reshape(14, 2)[:, 0]
				ys = poly_bbox.reshape(14, 2)[:, 1]

				# xs = oriented_box.reshape(4, 2)[:, 0]
				# ys = oriented_box.reshape(4, 2)[:, 1]
				xmin = xs.min()
				xmax = xs.max()
				ymin = ys.min()
				ymax = ys.max()
				bboxes.append([xmin, ymin, xmax, ymax])

				# might be wrong here, but it doesn't matter because the label is not going to be used in detection
				labels_text.append(str(config.text_label));
				labels.append(config.text_label)

			example = convert_to_example_ctw(image_data, image_name, labels, labels_text, bboxes, poly_bboxes, shape)
			tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":

	root_dir = '/home/xgbj/ctw1500/train/'
	output_dir = '/media/xgbj/9132EE0B9756C987/tf_record/pixel_link/ctw/'
	util.io.mkdir(output_dir);

	training_data_dir = util.io.join_path(root_dir, 'text_image')
	training_gt_dir = util.io.join_path(root_dir, 'text_label_curve')
	cvt_to_tfrecords(output_path=util.io.join_path(output_dir, 'ctw_train.tfrecord'), data_path=training_data_dir,
					 gt_path=training_gt_dir)

