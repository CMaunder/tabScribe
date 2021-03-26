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


def plot_spectrogram(y, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(
        y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")


class LearnNotes:

    def __init__(self):
        self.FRAME_SIZE = int(2048 * 2)
        self.HOP_SIZE = int(self.FRAME_SIZE / 16)
        self.FRAMES_PER_IMAGE = 1
        self.TEST_SET_SIZE = 0.1
        self.EPOCHS = 10

    def main(self):
        # todo - refactor
        resource_path = "./resources/allPitchNotes"
        data_dir = pathlib.Path(resource_path)
        note_names_unsorted = np.array(tf.io.gfile.listdir(str(data_dir)))
        note_names = np.sort(note_names_unsorted)
        data_images = []
        data_labels = []
        for note_name in note_names:
            curr_note_name = note_name
            dir_to_search = f'{resource_path}/{curr_note_name}'
            files_in_pitch_dir = glob.glob(f"{dir_to_search}/*.wav")
            highest_file_number = 0
            for file in files_in_pitch_dir:
                file_number = int(re.search("\\d+.wav", file).group()[:-4])
                if file_number > highest_file_number:
                    highest_file_number = file_number
            for i in range(1, highest_file_number + 1):
                filename = f'{dir_to_search}/{i}.wav'
                a, sr = librosa.load(filename)
                y_log_scale = self.convert_audio_to_spectrogram(a)
                min_mag = np.min(y_log_scale)
                y_log_scale = y_log_scale - min_mag
                max_mag = np.max(y_log_scale)
                y_log_scale = y_log_scale / max_mag
                note_names = np.array(tf.io.gfile.listdir(str(data_dir)))

                # plot_spectrogram(y_log_scale, sr, self.HOP_SIZE)
                # plt.show()

                # split into loads of smaller specs
                # the first 1 second is no note
                # the rest is the note as defined by the filename
                seconds_per_frame = 2.5 / len(y_log_scale[0])
                frame_of_note_start = 0.93 / seconds_per_frame

                for frame in range(len(y_log_scale[0])-self.FRAMES_PER_IMAGE):
                    note_name = 'na'
                    if frame > frame_of_note_start:
                        note_name = curr_note_name
                    data_images.append(y_log_scale[:, frame:frame + self.FRAMES_PER_IMAGE])
                    data_labels.append(note_name)

        note_names1 = np.insert(note_names, len(note_names), 'na')

        # now find random sample of exclude from train set and add to test set
        # todo - add validation split too
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
                raise Exception(f'Wrong image shape found: {str(train_image.shape)}')

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

        model.summary()

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(train_images, transformed_train_label, validation_data=(test_images, transformed_test_label),
                  epochs=self.EPOCHS, shuffle=True)

        test_loss, test_acc = model.evaluate(test_images, transformed_test_label)
        model.save("./models/notePredictModel3.h5")

        # model = keras.models.load_model('./models/notePredictModel1.h5')
        # test_loss, test_acc = model.evaluate(test_images, transformed_test_label)

        print(f"Train set size was: {len(train_images)}")
        print(f"Test set size was: {int(len(data_images) * self.TEST_SET_SIZE)}")
        print(f"Tested acc: {test_acc}")

        prediction = model.predict(test_images)

        y_pred = np.argmax(prediction, axis=1)
        y_true = []
        note_names1 = np.sort(note_names1)
        note_names2 = note_names1.tolist()
        for i in range(len(test_labels)):
            y_true.append(note_names2.index(test_labels[i]))
        print(len(y_true))
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=note_names1, yticklabels=note_names1,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        plt.show()

    def convert_audio_to_spectrogram(self, audio_sample):
        stft_audio = librosa.stft(audio_sample, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        y_scale = np.abs(stft_audio) ** 2
        y_log_scale = librosa.power_to_db(y_scale)
        # highest note on a guitar is d6 so anything much over 1175hz can be ignored
        # lowest note on a guitar is e2 which is 82 hz so anything much under 82 hz can be ignored
        return y_log_scale


learnNotes = LearnNotes()
learnNotes.main()
