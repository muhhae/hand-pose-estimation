import cv2
import os
import mediapipe as mp

os.environ["QT_QPA_PLATFORM"] = "xcb"

mp_hands = mp.solutions.hands
hands_estimator = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def findFingers(frame: cv2.UMat) -> cv2.UMat:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_estimator.process(img_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if ret is None:
            return

        frame = findFingers(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
