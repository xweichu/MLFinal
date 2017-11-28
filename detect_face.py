import cv2


def detect_faces(image):
    FACE_CASCADE = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)
    return faces


image = cv2.imread('people.jpg')
faces = detect_faces(image)
for x, y, w, h in faces:
    cv2.imshow("Face", image[y - 10:y + h + 10, x - 10:x + w + 10])
    cv2.waitKey()


