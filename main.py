# import OpenCV
import cv2

def grayscale(img):
    """Apply a grayscale filter to an image"""

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detectFacesWithRectangle(img, faces):
    """Draw a rectangle around faces found"""

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

# read the image
img = cv2.imread('barack_obama.jpg')

# load cascade classifier training file for haarcascade
face_cascade = cv2.CascadeClassifier('venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

# detect faces...
faces = face_cascade.detectMultiScale(grayscale(img), 1.1, 5)
print('Faces found: ', len(faces))

# highlight faces on the original image with rectangles
detectFacesWithRectangle(img, faces)

# output the image with facial recognition
cv2.imshow('Barack Obama', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



