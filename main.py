import cv2
import numpy as np

# FIRE HSV COLOR (from gbt)
fire_color_lower = np.array([5, 150, 150])
fire_color_upper = np.array([35, 255, 255])


def video(frame):
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # convert every frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower < pixel < upper --> white
        mask = cv2.inRange(hsv, fire_color_lower, fire_color_upper)

        # application filtres morphologique ( dilatation + erosion )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # contours search
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # drawing a box over flames
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # displaying the resultant video
        cv2.imshow('Test with a video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def image(image):
    image = cv2.imread(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, fire_color_lower, fire_color_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Test with an image', image)
    cv2.waitKey(0)


# give the user the choice
choice = input('lunch the fire detection on:\n 1: live video \n 2: existant video \n 3: image\n')

if choice == '1':
    cap = cv2.VideoCapture(0)
    video(cap)
    cap.release()
    cv2.destroyAllWindows()
elif choice == '2':
    cap = cv2.VideoCapture('ForestFire1.mp4')
    video(cap)
    cap.release()
    cv2.destroyAllWindows()
elif choice == '3':
    image('firepic.png')

