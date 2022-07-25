import cv2

webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("face-cascade.xml")
nose_cascade = cv2.CascadeClassifier("nose-cascade.xml")

center_coordinates = (120, 100)
axesLength = (100, 50)
angle = 0
startAngle = 0
endAngle = 360
# Red color in BGR
color = (0, 0, 255)

# Line thickness of 5 px
thickness = 5
print("\n\nPress 'esc' to quit program\n\n")
while (True):
    s, video = webcam.read()

    video = cv2.flip(video, 180)

    faces = face_cascade.detectMultiScale(video,
                                          minNeighbors=20,
                                          minSize=(30, 30),
                                          maxSize=(400, 400))
    # print(faces)
    for (x, y, w, h) in faces:
        # cv2.circle(video, (x, y), (x+w, y+h), (0, 255, 0), 4)
        # cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(video, (x + (int(w / 2)), y + (int(h / 2))), (round(w*0.45), round(h*0.6)), angle,
                    startAngle, endAngle, (0, 255, 0), 4)
        window_name = 'Image'

    nose = nose_cascade.detectMultiScale(video,
                                         minNeighbors=10,
                                         minSize=(10, 10),
                                         maxSize=(400, 400))
    for (x, y, w, h) in nose:
        # cv2.circle(video, (x, y), (x+w, y+h), (0, 255, 0), 4)
        # cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
        cv2.ellipse(video, (x+(int(w/2)), y+(int(h/2))), (round(w/2), round(h/2)), angle, startAngle, endAngle,
                    (0, 255, 0), 4)
        window_name = 'Image'


    cv2.imshow("Face Detection", video)

    if (cv2.waitKey(1) == 27):
        break
webcam.release()
cv2.destroyAllWindows()
