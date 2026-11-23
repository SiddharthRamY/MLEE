import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time
from spotify_control import execute_command


# -----------------------------
# Settings
# -----------------------------

SAMPLE_RATE = 16000

WINDOW_SIZE = 16000       # 1 second
STEP_SIZE = 4000          # 250 ms

THRESHOLD = 0.90

REQUIRED_DETECTIONS = 2

COOLDOWN = 1.5            # seconds

classes = [
    "up",
    "down",
    "go",
    "stop",
    "unknown",
    "silence"
]


# -----------------------------
# Load trained model
# -----------------------------

model = tf.keras.models.load_model("spotify_kws.keras")

print("Model loaded successfully.")
print("Starting microphone...")
print("Say: up / down / go / stop")
print("Press Ctrl+C to stop.\n")


# -----------------------------
# Audio buffer
# -----------------------------

audio_buffer = np.zeros(
    WINDOW_SIZE,
    dtype=np.float32
)


# -----------------------------
# Debounce variables
# -----------------------------

last_prediction = None
consecutive_count = 0

last_command_time = 0


# -----------------------------
# Continuous microphone
# -----------------------------

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=STEP_SIZE
) as stream:

    while True:

        # -----------------------------
        # Read new 250 ms of audio
        # -----------------------------

        new_audio, overflowed = stream.read(STEP_SIZE)

        new_audio = new_audio.flatten()


        # -----------------------------
        # Update sliding window
        # -----------------------------

        audio_buffer = np.concatenate(
            (
                audio_buffer[STEP_SIZE:],
                new_audio
            )
        )


        # -----------------------------
        # Extract MFCC
        # -----------------------------

        mfcc = librosa.feature.mfcc(
            y=audio_buffer,
            sr=SAMPLE_RATE,
            n_mfcc=13,
            n_fft=400,
            hop_length=160,
            win_length=400
        )


        # -----------------------------
        # Prepare CNN input
        # -----------------------------

        mfcc = mfcc[
            np.newaxis,
            ...,
            np.newaxis
        ]


        # -----------------------------
        # Prediction
        # -----------------------------

        probabilities = model.predict(
            mfcc,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_class = classes[predicted_index]

        confidence = probabilities[predicted_index]


        # -----------------------------
        # Check cooldown
        # -----------------------------

        current_time = time.time()

        if current_time - last_command_time < COOLDOWN:

            remaining = (
                COOLDOWN -
                (current_time - last_command_time)
            )

            print(
                f"Cooldown: {remaining:.1f}s"
            )

            continue


        # -----------------------------
        # Check if valid command
        # -----------------------------

        if (
            predicted_index <= 3
            and confidence >= THRESHOLD
        ):

            # Same command as previous prediction
            if predicted_class == last_prediction:

                consecutive_count += 1

            else:

                last_prediction = predicted_class
                consecutive_count = 1


            print(
                f"{predicted_class} "
                f"| {confidence:.2f} "
                f"| detections: "
                f"{consecutive_count}/{REQUIRED_DETECTIONS}"
            )
           


            # -----------------------------
            # Confirm command
            # -----------------------------

            if consecutive_count >= REQUIRED_DETECTIONS:

                print(
                    f"\n>>> COMMAND CONFIRMED: "
                    f"{predicted_class}"
                )
                execute_command(predicted_class)
                last_command_time = current_time

                # Reset debounce
                last_prediction = None
                consecutive_count = 0


        else:

            # Reset debounce if prediction
            # is not a valid command

            last_prediction = None
            consecutive_count = 0

            print(
                f"Ignored: {predicted_class} "
                f"| {confidence:.2f}"
            )