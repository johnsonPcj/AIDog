# USAGE
# python classify.py --model output/Garbage.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle \
#	--image examples/2.jpg

# python classify.py --model output/Garbage.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle \
#	--image examples/2.jpg
# or:python classify.py -m output/Garbage.model -l output/category_lb.pickle -c output/color_lb.pickle\
#	--image examples/2.jpg

"""in windows OS,classify.py CLI need file directory:
D:\mylab\cv\FruitNet>python classify.py -m output/Garbage.model -l output/category_lb.pickle -c output/color_lb.pickle \
-i examples/2.jpg
"""
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--categorybin", required=True,
	help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	help="path to output color label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = cv2.resize(image, (200,200))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# pre-process the image for classification
image = cv2.resize(image, (50, 50))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
categoryLB = pickle.loads(open(args["categorybin"], "rb").read())
colorLB = pickle.loads(open(args["colorbin"], "rb").read())

# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(categoryProba, colorProba) = model.predict(image)

# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels
categoryIdx = categoryProba[0].argmax()
colorIdx = colorProba[0].argmax()
categoryLabel = categoryLB.classes_[categoryIdx]
colorLabel = colorLB.classes_[colorIdx]

# draw the category label and color label on the image
categoryText = "category: {} ({:.2f}%)".format(categoryLabel,
	categoryProba[0][categoryIdx] * 100)
colorText = "color: {} ({:.2f}%)".format(colorLabel,
	colorProba[0][colorIdx] * 100)
cv2.putText(output, categoryText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.5, (255,0, 0), 2)
cv2.putText(output, colorText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
	0.5, (255,0, 0), 2)

# display the predictions to the terminal as well
print("[INFO] {}".format(categoryText))
print("[INFO] {}".format(colorText))

# show the output image
print(output.shape)

print(np.max(output))
print(np.min(output))


cv2.imshow("output", output)
cv2.waitKey()#3000