import pandas
import random
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2 as cv
import sys

test_image_count = 8
epochs_count = 30
calculate_epochs = False

print("DMX5314 - Case Study-01 - 116967442")

print('\nTrain or Test:')
print('\t Test  -0')
print('\t Train -1')
print('\nEnter your choice:')
choice1 = input()

# Decide if to load an existing model (Test) or to train a new one (Train)

if choice1 == '0':  # Test From Model
    train_new_model = False  # Set model train
    print('\nNumber of Test Images:')
    test_image_count = int(input())
elif choice1 == '1':  # Train The Model
    train_new_model = True  # Set model train
    print('\nCalculate Optimum epochs:')
    print('\t No  -0')
    print('\t Yes -1')
    choice2 = input()
    if choice2 == '0':  # Use Calculated Optimum epochs
        print('\nNumber of epochs:')
        epochs_count = int(input())
    elif choice2 == '1':  # Calculate Optimum epochs
        calculate_epochs = True
        epochs_count = 4
    else:
        print('\nInvalid Input:')
        sys.exit()
else:
    print('\nInvalid Input!')
    sys.exit()

data_path = r"D:\Acadamic OUSL\Level 5\Vision\Mini Project Code"

dataset = pandas.read_csv(r'D:\Acadamic OUSL\Level 5\Vision\Mini Project Code\english.csv')
rand = random.sample(range(len(dataset)), 500)
validation_set = pandas.DataFrame(dataset.iloc[rand, :].values, columns=['image', 'label'])
# remove the added data
dataset.drop(rand, inplace=True)

rand = random.sample(range(len(validation_set)), 5)
# remove the added data
validation_set.drop(rand, inplace=True)

train_data_generator = ImageDataGenerator(rescale=1 / 255, shear_range=0.2, zoom_range=0.2)
data_generator = ImageDataGenerator(rescale=1 / 255)
training_data_frame = train_data_generator.flow_from_dataframe(dataframe=dataset, directory=data_path, x_col='image',
                                                               y_col='label', target_size=(64, 64),
                                                               class_mode='categorical')
validation_data_frame = data_generator.flow_from_dataframe(dataframe=validation_set, directory=data_path, x_col='image',
                                                           y_col='label', target_size=(64, 64),
                                                           class_mode='categorical')

if train_new_model:

    model = tf.keras.models.Sequential()

    # add convolutional and pooling layer
    model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Flatten())

    # add full connection, output layer
    model.add(tf.keras.layers.Dense(units=600, activation='relu'))
    model.add(tf.keras.layers.Dense(units=52, activation='sigmoid'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if calculate_epochs:
        history = model.fit(training_data_frame, validation_data=validation_data_frame, epochs=epochs_count)

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs_count)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    else:
        model.fit(training_data_frame, validation_data=validation_data_frame, epochs=epochs_count)

    # Saving the model
    model.save('handwritten_digits.model')
    print('\nModel Trained Successfully')

else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model')

    f_data_path = r"D:\Acadamic OUSL\Level 5\Vision\Mini Project Code\Drop_test_Image_Here"

    f_dataset = pandas.read_csv(r'D:\Acadamic OUSL\Level 5\Vision\Mini Project Code\Drop_test_Image_Here\english.csv')
    f_rand = random.sample(range(len(f_dataset)), test_image_count)
    f_validation_set = pandas.DataFrame(f_dataset.iloc[f_rand, :].values, columns=['image', 'label'])
    # remove the added data
    f_dataset.drop(f_rand, inplace=True)

    f_rand = random.sample(range(len(f_validation_set)), test_image_count)
    f_test_set = pandas.DataFrame(f_validation_set.iloc[f_rand, :].values, columns=['image', 'label'])
    # remove the added data
    f_validation_set.drop(f_rand, inplace=True)

    print(f_test_set)

    f_data_generator = ImageDataGenerator(rescale=1 / 255)
    f_test_data_frame = f_data_generator.flow_from_dataframe(dataframe=f_test_set, directory=f_data_path, x_col='image',
                                                             y_col='label', target_size=(64, 64),
                                                             class_mode='categorical', shuffle=False)
    f_training_data_frame = train_data_generator.flow_from_dataframe(dataframe=dataset, directory=data_path,
                                                                     x_col='image', y_col='label', target_size=(64, 64),
                                                                     class_mode='categorical')

    print("F Prediction mapping: ", f_training_data_frame.class_indices)
    predict = model.predict(f_test_data_frame)

    # switcher shows our network mapping to the prediction
    switcher = {
        0: "A",   1: "B",  2: "C",  3: "D",  4: "E",  5: "F",  6: "G",  7: "H",  8: "I",  9: "J",
        10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
        20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "a", 27: "b", 28: "c", 29: "d",
        30: "e", 31: "f", 32: "g", 33: "h", 34: "i", 35: "j", 36: "k", 37: "l", 38: "m", 39: "n",
        40: "o", 41: "p", 42: "q", 43: "r", 44: "s", 45: "t", 46: "u", 47: "v", 48: "w", 49: "x",
        50: "y", 51: "z"}

    outputDf = pandas.DataFrame(predict)

    maxIndex = list(outputDf.idxmax(axis=1))

    for i in range(len(f_test_set)):
        image = img.imread(f_data_path + '/' + f_test_set.at[i, 'image'])
        plt.title(switcher.get(maxIndex[i], "error"))
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.show()
