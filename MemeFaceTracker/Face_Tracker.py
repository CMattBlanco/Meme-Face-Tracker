"""Meme Face Tracker - all memes in one file.   Press q in the camera window to quit.
Needs: pip install mediapipe opencv-python
"""
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/{0}/{0}/float16/1/{0}.task'

# ---- Thresholds. Lower = triggers more easily. ----
SMILE_AT = 0.5            # smile.jpg
FROWN_AT = 0.3            # sad.jpg / willie.jpg: mouth corners pulled down
BROWS_DOWN_AT = 0.5       #   ...or brows pulled down
EYES_CLOSED_AT = 0.5      # cute.jpg / willie.jpg
POUT_AT = 0.3             # cute.jpg
MOUTH_CLOSED_BELOW = 0.15 # sad.jpg
MOUTH_OPEN_AT = 0.25      # willie.jpg
SHOCK_MOUTH_AT = 0.4      # shock.jpg
EYES_WIDE_AT = 0.25       # shock.jpg
TONGUE_MOUTH_AT = 0.15    # Tongue_Out.jpg
TONGUE_REDNESS_AT = 10    # Tongue_Out.jpg: how much redder than cheeks

# Which image to show, checked top to bottom (first match wins)
MEMES = ['pakyu', 'shock', 'tongue', 'willie', 'cute', 'sad', 'smile']
FILES = {'pakyu': 'pakyu.jpg', 'shock': 'shock.jpg', 'tongue': 'Tongue_Out.jpg',
         'willie': 'willie.jpg', 'cute': 'Cute.jpg', 'sad': 'sad.jpg', 'smile': 'smile.jpg'}


def get_model(name):
    path = os.path.join(BASE, name + '.task')
    if not os.path.exists(path):
        print('Downloading', name, 'model...')
        urllib.request.urlretrieve(MODEL_URL.format(name), path)
    with open(path, 'rb') as f:
        return f.read()


# ---------- Load memes ----------
memes = {}
for key, fname in FILES.items():
    img = cv2.imread(os.path.join(BASE, fname))
    if img is None:
        print('WARNING: could not load', fname)
    else:
        memes[key] = cv2.resize(img, (480, 480))

# ---------- Trackers ----------
face_tracker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=get_model('face_landmarker')),
    running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True, num_faces=1))

try:
    hand_tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=get_model('hand_landmarker')),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
except Exception as e:
    print('WARNING: hand model unavailable, pakyu and shock disabled:', e)
    hand_tracker = None


# ---------- Helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(hand, tip, pip):
    """Finger is up when its tip is farther from the wrist than its middle joint."""
    return dist(hand[tip], hand[0]) > dist(hand[pip], hand[0])


def is_middle_finger(hand):
    return (finger_up(hand, 12, 10) and not finger_up(hand, 8, 6)
            and not finger_up(hand, 16, 14) and not finger_up(hand, 20, 18))


def hand_on_head(hand, face):
    """Middle of the hand is above the nose and around the face."""
    hx = sum(p.x for p in hand) / len(hand)
    hy = sum(p.y for p in hand) / len(hand)
    xs, ys = [p.x for p in face], [p.y for p in face]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    return (min(xs) - 0.7 * fw < hx < max(xs) + 0.7 * fw
            and min(ys) - 0.8 * fh < hy < face[1].y)


def redness(lab, x0, y0, x1, y1):
    """Average redness (Lab a-channel) of a box given in 0-1 coordinates."""
    h, w = lab.shape[:2]
    x0, x1 = sorted((int(x0 * w), int(x1 * w)))
    y0, y1 = sorted((int(y0 * h), int(y1 * h)))
    box = lab[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1, 1]
    return float(box.mean()) if box.size else 0.0


def tongue_redness(frame, L):
    """How much redder the area below the lower lip is than the cheeks."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    cx = (L[61].x + L[291].x) / 2
    half = abs(L[291].x - L[61].x) * 0.25
    chin = L[152].y - L[17].y
    below_lip = redness(lab, cx - half, L[17].y + chin * 0.05, cx + half, L[17].y + chin * 0.45)
    cheeks = (redness(lab, L[205].x - .02, L[205].y - .02, L[205].x + .02, L[205].y + .02) +
              redness(lab, L[425].x - .02, L[425].y - .02, L[425].x + .02, L[425].y + .02)) / 2
    return below_lip - cheeks


# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
start = time.time()
shown = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = int((time.time() - start) * 1000)

    hands = []
    if hand_tracker:
        hands = hand_tracker.detect_for_video(image, t).hand_landmarks
    face = face_tracker.detect_for_video(image, t)

    triggers = {m: False for m in MEMES}
    info = 'no face'

    triggers['pakyu'] = any(is_middle_finger(h) for h in hands)

    if face.face_blendshapes:
        s = {b.category_name: b.score for b in face.face_blendshapes[0]}
        both = lambda name: (s[name + 'Left'] + s[name + 'Right']) / 2   # left/right average
        L = face.face_landmarks[0]

        jaw = s['jawOpen']
        smile = both('mouthSmile')
        eyes = both('eyeBlink')
        wide = both('eyeWide')
        pout = s['mouthPucker']
        frowning = both('mouthFrown') > FROWN_AT or both('browDown') > BROWS_DOWN_AT
        tongue = tongue_redness(frame, L)
        on_head = sum(hand_on_head(h, L) for h in hands)

        triggers['shock'] = jaw > SHOCK_MOUTH_AT and wide > EYES_WIDE_AT and on_head >= 2
        triggers['tongue'] = jaw > TONGUE_MOUTH_AT and tongue > TONGUE_REDNESS_AT
        triggers['willie'] = frowning and eyes > EYES_CLOSED_AT and jaw > MOUTH_OPEN_AT
        triggers['cute'] = eyes > EYES_CLOSED_AT and pout > POUT_AT
        triggers['sad'] = frowning and jaw < MOUTH_CLOSED_BELOW
        triggers['smile'] = smile > SMILE_AT

        info = (f'smile:{smile:.2f} jaw:{jaw:.2f} eyes:{eyes:.2f} wide:{wide:.2f} '
                f'pout:{pout:.2f} tongue:{tongue:.1f} hands-on-head:{on_head}')

    expression = next((m for m in MEMES if triggers[m] and m in memes), None)

    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, 'meme: ' + (expression or '-'), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow('Camera', frame)

    if expression:
        cv2.imshow('Meme', memes[expression])
        shown = expression
    elif shown:
        cv2.destroyWindow('Meme')
        shown = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
