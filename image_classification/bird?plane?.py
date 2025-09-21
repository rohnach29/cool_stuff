import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#want to scale down values so that they're between 0 and 1.

#tried this: testing_images /= 255 doesn't work!! because it's in place division and the arrays are currently uint8 -- = ... / creates a new array

training_images = training_images / 255
testing_images = testing_images / 255
print(training_labels, '\n', testing_labels)

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

#just print out the first 16 images
# for i in range(16):
#     plt.subplot(4, 4, i + 1)    #4x4 grid of images
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i])#, cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])

# plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]

testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# #need to understand this!!! keras seems so chill, but also need to build the intuition of how to build this. i know that CNN -> image classification but don't know what layers to add
# model = models.Sequential()
# #32 filters, each 3x3. images are 32x32. relu: negative → 0 isn’t arbitrary. It’s like saying “if I didn’t detect my feature, don’t clutter the signal with noise.”
# #common practice is to use relu for CNNs. why not sigmoid/tanh? for deep networks, these vanish gradients -> largely +ve or -ve numbers, slope is almost 0.
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (32, 32, 3)))   #output shape: 30x30x32. 32 filters
# #pooling: take every 2x2 square and take its maximum value (or average in some cases, or minimum). default stride is 2
# #why do this? less parameters. if 32x32, we don't need to learn EXACTLY where the dog's ear starts, just that is there somewhere in that region.
# model.add(layers.MaxPooling2D((2,2)))   #15x15x32 here
# model.add(layers.Conv2D(64, (3,3), activation='relu'))  #64 here -> more filters to learn more complex patterns. same (3x3), but increasing filters to increase complexitu
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))  #64 again? enough capacity for this dataset. not too sure about how this works right now and what the design intuition is, i'll look into it more
# #it is like 4x4x64 here.
# model.add(layers.Flatten())     #flatten it to feed into a Dense layers. conv + pooling learns local spatial features. to classify into categories, need a single feature vector summarizing the whole image
# #what is the intuition behind this? why 3 layers of conv2D and (3,3)? more of a design question that we can look into later
# model.add(layers.Dense(64, activation='relu'))  #64 is a hyperparameter. this layers looks at all raw features and combines them into higher-level concepts
# #final dense layer with softmax activation for probabilities!
# model.add(layers.Dense(10, activation='softmax'))

# #can learn more about adam and RMSProp. 
# #categorical_crossentropy -> one hot vectors. sparse -> not one hot vectors. labels from 1....9
# #reminder: categorical_crossentropy -> L=−log(probability assigned to the true class). overall loss -> average over whole dataset.
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)

# print(f"accuracy: {accuracy}")

# print(f"loss: {loss}")

# model.save("image_classifier.keras")

model = models.load_model("image_classifier.keras")

#now going to take images from the internet and classify them with this model
img = cv.imread("car.jpg")
#creates as BGR -> we need to make it RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img]) / 255)

index = np.argmax(prediction)
print(f"This is an image of a {class_names[index]}")