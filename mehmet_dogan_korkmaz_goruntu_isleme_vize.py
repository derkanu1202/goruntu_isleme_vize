from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import random

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Başlangıçta bir çöpün konumunu belirle
trash_x = random.randint(100, 500)
trash_y = random.randint(100, 400)
score = 0

def koordinat_getir(landmarks, indeks, h, w):
  landmark = landmarks[indeks]
  return int(landmark.x*w), int(landmark.y*h)

def draw_landmarks_on_image(rgb_image, detection_result):
  global trash_x, trash_y, score

  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  h, w, c = annotated_image.shape

  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)  # İşaret parmak ucu

    # Parmağın ucuna daire çiz
    annotated_image = cv2.circle(annotated_image, (x1, y1), 10, (255, 0, 0), -1)

    # Çöp ile çakışma kontrolü
    mesafe = np.hypot(trash_x - x1, trash_y - y1)
    if mesafe < 40:
      score += 1
      trash_x = random.randint(100, 500)
      trash_y = random.randint(100, 400)

    # El iskeletini çiz
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # El yönünü yazdır
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * w)
    text_y = int(min(y_coordinates) * h) - MARGIN
    cv2.putText(annotated_image, f"{handedness_list[idx][0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  # Çöpü çiz
  annotated_image = cv2.circle(annotated_image, (trash_x, trash_y), 20, (0, 255, 0), -1)
  # Skoru göster
  annotated_image = cv2.putText(annotated_image, f"Skor: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  return annotated_image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      detection_result = detector.detect(mp_image)

      annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      cv2.imshow("Cöp Toplama Oyunu", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

      key = cv2.waitKey(1)
      if key == ord('q') or key == ord('Q'):
         break

cam.release()
cv2.destroyAllWindows()
