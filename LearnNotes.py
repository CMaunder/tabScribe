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
        self.TEST_VAL_SET_SIZE = 0.3
        self.EPOCHS = 5

    def main(self):
        model_name = "notePredictModel5"
        self.generate_and_save_model(model_name)
        # model = keras.models.load_model('./models/notePredictModel1.h5')

    def generate_and_save_model(self, model_name):
        resource_path = "./resources/allPitchNotes"
        [data_images, data_labels, note_names] = self.extract_data_from_files(resource_path)
        # Find random sample of exclude from train set and add to test/val set
        indices_of_test_val_set = np.random.randint(len(data_images),
                                                    size=int(len(data_images) * self.TEST_VAL_SET_SIZE))
        # Split val and test data in 50:50 ratio
        indices_of_test_set = indices_of_test_val_set[:int(len(indices_of_test_val_set) / 2)]
        indices_of_val_set = indices_of_test_val_set[int(len(indices_of_test_val_set) / 2):]
        train_images = data_images
        train_labels = data_labels
        test_images = []
        test_labels = []
        val_images = []
        val_labels = []
        for index in indices_of_test_set:
            test_images.append(data_images[index])
            test_labels.append(data_labels[index])
        for index in indices_of_val_set:
            val_images.append(data_images[index])
            val_labels.append(data_labels[index])
        for index in sorted(indices_of_test_set, reverse=True):
            del train_images[index]
            del train_labels[index]
        print(f"Size of train data set is: {len(train_images)}")
        print(f"Size of validation data set is: {len(val_images)}")
        print(f"Size of test data set is: {len(test_images)}")
        train_images = np.array(train_images)
        test_images = np.array(test_images)
        val_images = np.array(val_images)
        train_images = np.asarray(train_images)
        train_labels = np.asarray(train_labels)
        test_images = np.asarray(test_images)
        test_labels = np.asarray(test_labels)
        val_images = np.asarray(val_images)
        val_labels = np.asarray(val_labels)

        # Convert text labels into ones and zeros
        encoder = LabelBinarizer()
        transformed_train_label = encoder.fit_transform(train_labels)
        transformed_test_label = encoder.fit_transform(test_labels)
        transformed_val_labels = encoder.fit_transform(val_labels)

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(len(test_images[0]), self.FRAMES_PER_IMAGE)),
            keras.layers.Dense(128*4, activation="relu"),
            keras.layers.Dense(len(note_names), activation="softmax")
        ])
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_images, transformed_train_label,
                  validation_data=(val_images, transformed_val_labels),
                  epochs=self.EPOCHS, shuffle=True,
                  callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=20))

        test_loss, test_acc = model.evaluate(test_images, transformed_test_label)
        model.save(f"./models/{model_name}.h5")
        print(f"Train set size was: {len(train_images)}")
        print(f"Test set size was: {len(test_images)}")
        print(f"Tested acc: {test_acc}")
        y_prediction = np.argmax(model.predict(test_images), axis=1)
        y_true = []
        note_names_sorted = np.sort(note_names)
        note_names_list = note_names_sorted.tolist()
        for i in range(len(test_labels)):
            y_true.append(note_names_list.index(test_labels[i]))
        confusion_mtx = tf.math.confusion_matrix(y_true, y_prediction)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=note_names_sorted, yticklabels=note_names_sorted,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.show()

    def convert_audio_to_spectrogram(self, audio_sample):
        stft_audio = librosa.stft(audio_sample, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        y_scale = np.abs(stft_audio) ** 2
        y_log_scale = librosa.power_to_db(y_scale)
        return y_log_scale

    def extract_data_from_files(self, resource_path):
        print("Extracting and fragmenting data from files...")
        data_dir = pathlib.Path(resource_path)
        note_names_unsorted = np.array(tf.io.gfile.listdir(str(data_dir)))
        note_names = np.sort(note_names_unsorted)
        data_images = []
        data_labels = []
        for note_name in note_names:
            print(f"Extracting data for note: {note_name}")
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
                seconds_per_frame = 2.5 / len(y_log_scale[0])
                frame_of_note_start = 0.93 / seconds_per_frame

                for frame in range(len(y_log_scale[0]) - self.FRAMES_PER_IMAGE):
                    note_name = 'na'
                    if frame > frame_of_note_start:
                        note_name = curr_note_name
                    data_images.append(y_log_scale[:, frame:frame + self.FRAMES_PER_IMAGE])
                    data_labels.append(note_name)

        note_names1 = np.insert(note_names, len(note_names), 'na')
        for train_image in data_images:
            if str(train_image.shape) != f'(2049, {self.FRAMES_PER_IMAGE})':
                raise Exception(f'Wrong image shape found: {str(train_image.shape)}')
        print("Finished extracting and fragmenting data from files")
        return [data_images, data_labels, note_names1]


learnNotes = LearnNotes()
learnNotes.main()
