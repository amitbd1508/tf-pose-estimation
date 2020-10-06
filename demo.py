import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def image_resize(img, width=None, height=None, inter=cv2.INTER_AREA):
  # initialize the dimensions of the image to be resized and
  # grab the image size
  dim = None
  (h, w) = img.shape[:2]

  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
    return img

  # check to see if the width is None
  if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    dim = (int(w * r), height)

  # otherwise, the height is None
  else:
    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

  # resize the image
  resized = cv2.resize(img, dim, interpolation=inter)

  # return the resized image
  return resized


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
  parser.add_argument('--src', type=int, default=0, help='Video path or camera source like 0,1,2')

  parser.add_argument('--resize', type=str, default='0x0',
                      help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
  parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                      help='if provided, resize heatmaps before they are post-processed. default=1.0')

  parser.add_argument('--model', type=str, default='mobilenet_thin',
                      help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
  parser.add_argument('--show-process', type=bool, default=False,
                      help='for debug purpose, if enabled, speed for inference is dropped.')

  parser.add_argument('--tensorrt', type=str, default="False",
                      help='for tensorrt process.')

  parser.add_argument('--thickness', type=int, default=15,
                      help='Specify the thickness')

  args = parser.parse_args()

  logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
  w, h = model_wh(args.resize)
  if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
  else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
  cam = cv2.VideoCapture(args.src)
  ret_val, image = cam.read()
  logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

  while True:
    ret_val, image = cam.read()

    logger.debug('image process+')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

    logger.debug('postprocess+', humans)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, thikness=args.thickness)

    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('Human Pose Detection Demo', image_resize(image, height = 640))
    fps_time = time.time()
    if cv2.waitKey(1) == 27:
      break
    logger.debug('finished+')

  cv2.destroyAllWindows()
