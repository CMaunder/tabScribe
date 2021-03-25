import pathlib

import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import glob
import re


class LearnNotes:

    def __init__(self):
        self.FRAME_SIZE = int(2048 * 2)
        self.HOP_SIZE = int(self.FRAME_SIZE / 16)
        self.FRAMES_PER_IMAGE = 1
        self.TEST_SET_SIZE = 0.1
        self.EPOCHS = 5

    def main(self):
        # todo - refactor
        resource_path = "./resources/allPitchNotes"
        data_dir = pathlib.Path(resource_path)
        note_names = np.array(tf.io.gfile.listdir(str(data_dir)))
        data_images = []
        data_labels = []
        for note_name in note_names:
            curr_note_name = note_name
            dir_to_search = f'{resource_path}/{curr_note_name}'
            filesInPitchDir = glob.glob(f"{dir_to_search}/*.wav")
            highestFileNumber = 0
            for file in filesInPitchDir:
                fileNumber = int(re.search("\\d+.wav", file).group()[:-4])
                if fileNumber > highestFileNumber:
                    highestFileNumber = fileNumber
            for i in range(1, highestFileNumber+1):
                filename = f'{dir_to_search}/{i}.wav'
                a, sr = librosa.load(filename)
                y_log_scale = self.convert_audio_to_spectrogram(a, sr)
                min_mag = np.min(y_log_scale)

                # tranform mags so min is zero
                y_log_scale = y_log_scale - min_mag
                # normalize to 1 as the max

                max_mag = np.max(y_log_scale)
                y_log_scale = y_log_scale / max_mag
                note_names = np.array(tf.io.gfile.listdir(str(data_dir)))

                # self.plot_spectrogram(y_log_scale, sr, self.HOP_SIZE)
                # plt.show()

                # split into loads of smaller specs
                # the first 1 second is no note
                # the rest is the note as defined by the filename
                seconds_per_frame = 2.5 / len(y_log_scale[0])
                frame_of_note_start = 0.93 / seconds_per_frame

                for j in range(len(y_log_scale[0])):
                    note_name = 'na'
                    if j > frame_of_note_start:
                        note_name = curr_note_name
                    # todo - prevent last shape being 1 row instead of frame_per_image
                    data_images.append(y_log_scale[:, j:j + self.FRAMES_PER_IMAGE])
                    data_labels.append(note_name)

        note_names1 = np.insert(note_names, len(note_names), 'na')

        # now find random sample of exclude from train set and add to test set
        indices_of_test_set = np.random.randint(len(data_images), size=int(len(data_images) * self.TEST_SET_SIZE))
        train_images = data_images
        train_labels = data_labels
        test_images = []
        test_labels = []

        for index in indices_of_test_set:
            test_images.append(data_images[index])
            test_labels.append(data_labels[index])
        for index in sorted(indices_of_test_set, reverse=True):
            del train_images[index]
            del train_labels[index]

        for train_image in train_images:
            if str(train_image.shape) != f'(2049, {self.FRAMES_PER_IMAGE})':
                raise Exception('Wrong image shape found.')

        train_images = np.array(train_images)
        test_images = np.array(test_images)

        train_images = np.asarray(train_images)
        train_labels = np.asarray(train_labels)
        test_images = np.asarray(test_images)
        test_labels = np.asarray(test_labels)

        encoder = LabelBinarizer()
        transformed_train_label = encoder.fit_transform(train_labels)
        transformed_test_label = encoder.fit_transform(test_labels)

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(len(test_images[0]), self.FRAMES_PER_IMAGE)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(len(note_names1), activation="softmax")
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(train_images, transformed_train_label, epochs=self.EPOCHS)

        test_loss, test_acc = model.evaluate(test_images, transformed_test_label)
        model.save("./models/notePredictModel3.h5")
        print(f"Train set size was: {len(train_images)}")
        print(f"Test set size was: {int(len(data_images)*self.TEST_SET_SIZE)}")
        print(f"Tested acc: {test_acc}")

        prediction = model.predict(test_images)

        y_pred = np.argmax(prediction, axis=1)
        y_true = []
        note_names2 = note_names1.tolist()
        for i in range(len(test_labels)):
            y_true.append(note_names2.index(test_labels[i]))

        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=note_names1, yticklabels=note_names1,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        plt.show()

    def convert_audio_to_spectrogram(self, audio_sample, sr):
        stft_audio = librosa.stft(audio_sample, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        Y_scale = np.abs(stft_audio) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)
        hzPerFreqBin = sr / self.FRAME_SIZE
        # highest note on a guitar is d6 so anything much over 1175hz can be ignored
        # lowest note on a guitar is e2 which is 82 hz so anything much under 82 hz can be ignored
        return Y_log_scale

    def plot_spectrogram(self, y, sr, hop_length, y_axis="log"):
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(
            y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
        plt.colorbar(format="%+2.f")


learnNotes = LearnNotes()
learnNotes.main()
