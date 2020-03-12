
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imageai.Detection import ObjectDetection
import os
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image



def get_color_dist(image):
  color = ('b','r','g')
  plt.figure()
  for i, col in enumerate(color):
    hist = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
  plt.title("color distribution")
  plt.show()


def get_size_and_common_colors(image_path, position):
    img = Image.open(image_path)
    im1 = img.crop((position))
    size = im1.size
    colors = im1.convert('RGB').getcolors(im1.size[0] * im1.size[1])
    df = pd.DataFrame(columns=["count", "value"])

    for i in range(len(colors)):
        df = df.append({"count": colors[i][0], "value": colors[i][1]}, ignore_index=True)

    # plt.imshow(im1)
    # plt.show()

    return size, df.sort_values(by='count', ascending=False).value.head(2)


def common_colors(image_path):
    img = Image.open(image_path)
    size = img.size
    colors = img.convert('RGB').getcolors(img.size[0] * img.size[1])
    df = pd.DataFrame(columns=["count", "value"])

    for i in range(len(colors)):
        df = df.append({"count": colors[i][0], "value": colors[i][1]}, ignore_index=True)

    # plt.imshow(im1)
    # plt.show()

    return size, df.sort_values(by='count', ascending=False).value.head(2)

def detect_object(image_name):
    detector = ObjectDetection()

    model_path = "/content/task/models/yolo-tiny.h5"
    input_path = "/content/task/ScreenShots/" + image_name
    output_path = "/content/task/output/new_" + image_name

    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)

    detector.loadModel()
    detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path,
                                                minimum_percentage_probability=20)
    return detection


if __name__ == "__main__":

    current_path=os.getcwd()
    img_directory = os.path.abspath(os.path.join(
    current_path,
    '..',
    '..',
    'src',
    'screen_shots_small'))

    for filename in os.listdir(img_directory):
      if filename.endswith(".png"):
        start = time.time()
        image = cv2.imread(os.path.join(img_directory, filename))
        get_color_dist(image)
        size, colors = common_colors(os.path.join(img_directory, filename))
        print( "object size: ", size,  "\n   main color: ", colors.values[0], "\n   second main color: ", colors.values[1])
