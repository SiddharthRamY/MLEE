import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd     
import kagglehub
import librosa
import os
from playsound3 import playsound
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import random



path = kagglehub.dataset_download(
    "yashdogra/speech-commands"
)

print(path)


print(os.listdir(path))

for root, dirs, files in os.walk(path):
    wav_files = []

    for f in files:
        if f.endswith(".wav"):
            wav_files.append(f)

    if wav_files:
        print("Folder:", root)
        print("Number of WAV files:", len(wav_files))
        print("Examples:", wav_files[:5])
classes = ["up", "down", "go", "stop", "unknown", "silence"]

UNKNOWN_WORDS = [
    "yes",
    "no",
    "left",
    "right",
    "on",
    "off",
    "cat",
    "dog",
    "happy",
    "house"
]

silence_data = []

noise_folder = os.path.join(path, "_background_noise_")

for file in os.listdir(noise_folder):
    if file.endswith(".wav"):
        file_path = os.path.join(noise_folder, file)

        audio, sr = librosa.load(
            file_path,
            sr=16000,
            mono=True
        )

        num_chunks = len(audio) // 16000

        for i in range(num_chunks):
            start = i * 16000
            end = start + 16000

            chunk = audio[start:end]

            mfcc = librosa.feature.mfcc(
                y=chunk,
                sr=16000,
                n_mfcc=13,
                n_fft=400,
                hop_length=160,
                win_length=400
            )

            silence_data.append(mfcc)
print("Number of silence examples:", len(silence_data))
print("Silence MFCC shape:", silence_data[0].shape)
random.seed(42)
random.shuffle(silence_data)

n = len(silence_data)

train_silence = silence_data[:int(0.8 * n)]
val_silence = silence_data[int(0.8 * n):int(0.9 * n)]
test_silence = silence_data[int(0.9 * n):]

for file in os.listdir(noise_folder):
    if file.endswith(".wav"):
        print(file)
# for c in classes:
#     folder = os.path.join(path, c)

#     print(c, len(os.listdir(folder)))



playsound(r"C:\Users\Acer\.cache\kagglehub\datasets\yashdogra\speech-commands\versions\1\up\00b01445_nohash_0.wav")

def extract_mfcc(file_path):


    audio, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    if len(audio)  < sr:
        audio = np.pad(audio, (0, sr - len(audio)), mode='constant')
    elif len(audio) > sr:
        audio = audio[:sr]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=400,
        hop_length=160,
        win_length=400
    )

    return mfcc
def load_split_list(filename):
    with open(os.path.join(path, filename), "r") as f:
        return set(line.strip() for line in f if line.strip())
validation_files = load_split_list("validation_list.txt")
testing_files = load_split_list("testing_list.txt")

def get_split(relative_path):
    if relative_path in validation_files:
        return "validation"
    elif relative_path in testing_files:
        return "testing"
    else:
        return "training"
def collect_files(words, label_index):
    data = []

    for word in words:
        folder = os.path.join(path, word)

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                relative_path = f"{word}/{file}"

                split = get_split(relative_path)

                data.append((relative_path, label_index, split))

    return data
command_files = []

for label_index, word in enumerate(["up", "down", "go", "stop"]):
    command_files.extend(
        collect_files([word], label_index)
    )
unknown_files = collect_files(UNKNOWN_WORDS, 4)

random.seed(42)

if len(unknown_files) > 4000:
    unknown_files = random.sample(unknown_files, 4000)

all_files = command_files + unknown_files

train_files = []
val_files = []
test_files = []

for file_path, label, split in all_files:
    if split == "training":
        train_files.append((file_path, label))
    elif split == "validation":
        val_files.append((file_path, label))
    elif split == "testing":
        test_files.append((file_path, label))


print("Total files:", len(all_files))

print("\nClass distribution:")

# class_counts = Counter(label for _, label, _ in all_files)

# for label_index, count in sorted(class_counts.items()):
#     print(classes[label_index], ":", count)

# print("\nSplit distribution:")

# split_counts = Counter(split for _, _, split in all_files)

# for split, count in split_counts.items():
#     print(split, ":", count)


mfcc = extract_mfcc(r"C:\Users\Acer\.cache\kagglehub\datasets\yashdogra\speech-commands\versions\1\up\00b01445_nohash_0.wav")
print ("MFCC shape:", mfcc.shape)


plt.figure(figsize=(10, 4))

plt.imshow(
    mfcc,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Time frames")
plt.ylabel("MFCC coefficient")
plt.title("MFCC - UP")

plt.colorbar()

plt.show()

X_train = []
y_train = []

X_val = []
y_val = []

X_test = []
y_test = []
for file_path, label in train_files:
    full_path = os.path.join(path, file_path)
    X_train.append(extract_mfcc(full_path))
    y_train.append(label)
for file_path, label in val_files:
    full_path = os.path.join(path, file_path)
    X_val.append(extract_mfcc(full_path))
    y_val.append(label)
for file_path, label in test_files:
    full_path = os.path.join(path, file_path)
    X_test.append(extract_mfcc(full_path))
    y_test.append(label)
X_train=X_train+train_silence
y_train=y_train+[5]*len(train_silence)
X_val=X_val+val_silence
y_val=y_val+[5]*len(val_silence)
X_test=X_test+test_silence
y_test=y_test+[5]*len(test_silence)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# print("\nTraining:")
# print(Counter(y_train))

# print("\nValidation:")
# print(Counter(y_val))

# print("\nTesting:")
# print(Counter(y_test))
# X = []
# y = []

# for label_index, label in enumerate(classes):

#     folder = os.path.join(path, label)

#     for filename in os.listdir(folder):

#         if not filename.endswith(".wav"):
#             continue

#         file_path = os.path.join(folder, filename)

#         mfcc = extract_mfcc(file_path)

#         X.append(mfcc)
#         y.append(label_index)
# X = np.array(X)
# y = np.array(y)

# print("X shape:", X.shape)
# print("y shape:", y.shape)

# print(classes)
# print(np.unique(y, return_counts=True))




# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# print("Training X:", X_train.shape)
# print("Testing X:", X_test.shape)

# print("Training y:", y_train.shape)
# print("Testing y:", y_test.shape)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_val shape:", X_val.shape)




model = models.Sequential([

    layers.Input(shape=X_train.shape[1:]),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        len(classes),
        activation="softmax"
    )
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)
from sklearn.metrics import classification_report

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print(classification_report(
    y_test,
    y_pred,
    target_names=classes
))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)

plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.colorbar()

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.show()

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
import numpy as np

thresholds = [0.90, 0.92, 0.94, 0.96, 0.98]

print("\nThreshold Analysis")
print("-" * 70)

for threshold in thresholds:

    max_prob = np.max(y_pred_prob, axis=1)
    predicted_class = np.argmax(y_pred_prob, axis=1)

    # Accept only predictions above threshold
    accepted = max_prob >= threshold

    # Actual Spotify commands
    actual_commands = (y_test >= 0) & (y_test <= 3)

    # Correctly recognized commands
    correct_commands = (
        accepted &
        actual_commands &
        (predicted_class == y_test)
    )

    # Commands that were recognized (regardless of whether correct)
    recognized_commands = accepted & actual_commands

    # Unknown/silence that triggered a command
    false_triggers = (
        accepted &
        (y_test >= 4) &
        (predicted_class <= 3)
    )

    command_count = np.sum(actual_commands)
    non_command_count = np.sum(y_test >= 4)

    command_recall = np.sum(correct_commands) / command_count

    false_trigger_rate = (
        np.sum(false_triggers) / non_command_count
    )

    accepted_percentage = np.mean(accepted)

    print(f"Threshold: {threshold:.2f}")
    print(f"  Accepted predictions: {accepted_percentage * 100:.2f}%")
    print(f"  Command recall:       {command_recall * 100:.2f}%")
    print(f"  False trigger rate:   {false_trigger_rate * 100:.2f}%")
    print()
model.save("spotify_kws.keras")