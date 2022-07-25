
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
testa = []
queixo = []
boca = []
points = [10,15,200]

def saveHistory(lms, obj, point):
    item = [round(lms.landmark[point].x*640), round(lms.landmark[point].y*480)]
    if(len(obj)<=10):
        obj.append(item)
    else:
        obj.pop(0)
        obj.append(item)
    # print(obj)
    return obj

def verificaMovimento(obj):
    movimento = [0,0,0,0]
    for i in range(len(obj)-1, 0, -1):
        # print("=> " + str(i))
        if(obj[i][1]<obj[i-1][1]):
            # cima
            movimento[0] = movimento[0]+1 
        if(obj[i][1]>obj[i-1][1]):
            # baixo
            movimento[1] = movimento[1]+1       
        if(obj[i][0]<obj[i-1][0]):
            # direita
            movimento[2] = movimento[2]+1
        if(obj[i][0]>obj[i-1][0]):
            # esquerda
            movimento[3] = movimento[3]+1
    
    max_value = None
    max_idx = None
    for idx, num in enumerate(movimento):
        if (max_value is None or num > max_value):
            max_value = num
            max_idx = idx
        if(max_value>(0.7*len(obj))):
            if(max_idx == 0):
                print("Virou pra cima!")
            elif(max_idx == 1):
                print("Virou pra baixo!")
            elif(max_idx == 2):
                print("Virou pra direita!")
            elif(max_idx == 3):
                print("Virou pra esquerda!")
            else:
                print("Indefinido!")
        # print(movimento)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        # print(len(face_landmarks))
        # testa: 10
        # boca 15
        # Queixo 200
        boca = saveHistory(face_landmarks, boca, points[1])
        verificaMovimento(boca)
        # print("X:" + str(round(face_landmarks.landmark[325].x*640)))
        # print("Y:" + str(round(face_landmarks.landmark[325].y*480)))
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

    
    