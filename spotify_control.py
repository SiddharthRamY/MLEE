import spotipy
from spotipy.oauth2 import SpotifyOAuth


# -----------------------------
# Spotify settings
# -----------------------------

CLIENT_ID = ""
CLIENT_SECRET = ""

REDIRECT_URI = "http://127.0.0.1:8888/callback"

SCOPE = (
    "user-modify-playback-state "
    "user-read-playback-state"
)


# -----------------------------
# Spotify authentication
# -----------------------------

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_cache"
    )
)


print("Spotify connected!")

def next_track():
    sp.next_track()
    print("⏭️ Next track")


def previous_track():
    sp.previous_track()
    print("⏮️ Previous track")


def play():
    sp.start_playback()
    print("▶️ Playback resumed")


def pause():
    sp.pause_playback()
    print("⏸️ Playback paused")


def execute_command(command):

    try:
        if command == "up":
            next_track()

        elif command == "down":
            previous_track()

        elif command == "go":
            play()

        elif command == "stop":
            pause()

    except Exception as e:
        print(f"Spotify error: {e}")