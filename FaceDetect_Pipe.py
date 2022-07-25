''' 
 Trabalho de processamento de imagens - UFMT - IC
 Alunos: Jadson Matheus Lima (20181190102) Sergio Lucio Nunes (201811901024)  
 Tema: Reconhecimento de rosto e deteccao de movimento nos 4 eixos. 
 
 Instalar bibliotecas: OpenCV e MediaPipe

'''
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

boca = []
points = [10,15,200]
center_coordinates = (320, 240)
axesLength = (170, 140) 
angle = 90
startAngle = 0
endAngle = 360
thickness = 3

limitY1 = 200
limitY2 = 370
limiarY1 = round(limitY1 * 0.5)
limiarY2 = round(limitY2 * 0.5)
limitX1 = 320
limitX2 = 320
limiarX1 = round(limitX1 * 0.2)
limiarX2 = round(limitX2 * 0.2)

# Funcao que salva ultimos pontos dos landmarks para verificar mudanca de posicao e movimento
def saveHistory(lms, obj, point):
    item = [round(lms.landmark[point].x*640), round(lms.landmark[point].y*480)]
    if(len(obj)<=10):
        obj.append(item)
    else:
        obj.pop(0)
        obj.append(item)
    # print(obj)
    return obj

# Funcao que verifica se rosto esta dentro da elipse
def verificaPosicao(lms, point):
    if (lms.landmark[point[0]].y * 480 <= limitY1 + limiarY1 and lms.landmark[point[0]].y * 480 >= limitY1 - limiarY1):
        if (lms.landmark[point[2]].y * 480 >= limitY2 - limiarY2 and lms.landmark[point[2]].y * 480 <= limitY2):
            if (lms.landmark[point[0]].x * 640 <= limitX1 + limiarX1 and lms.landmark[point[0]].x * 640 >= limitX1 - limiarX1):
                if (lms.landmark[point[2]].x * 640 <= limitX2 + limiarX2 and lms.landmark[point[2]].x * 640  >= limitX1 - limiarX1):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

#Funcao que desenha elipse com cor passada por parametro
def drawEllipse(color):
    cv2.ellipse(
        image,
        center_coordinates, 
        axesLength,
        angle, 
        startAngle, 
        endAngle, 
        color, 
        thickness
    )

# Funcao que avalia o historico salvo dos pontos observados e identifica movimento 
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
                return "Virou pra cima!"
            elif(max_idx == 1):
                print("Virou pra baixo!")
                return "Virou pra baixo!"
            elif(max_idx == 2):
                print("Virou pra direita!")
                return "Virou pra direita!"
            elif(max_idx == 3):
                print("Virou pra esquerda!")
                return "Virou pra esquerda!"
            else:
                print("Indefinido!")
                return "Indefinido!"
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
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.flip(image, 1)
    x, y, w, h = 10, 10, 300,60
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        if(verificaPosicao(face_landmarks, points)):
            boca = saveHistory(face_landmarks, boca, points[1])
            cv2.rectangle(
                image, 
                (x, x), 
                (x + w + 40, y + h - 10), 
                (0,0,0), -1
            )
            cv2.putText(
                image, 
                verificaMovimento(boca), 
                (x + int(w/10),y + int(h/1.5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0)
            )
            drawEllipse((0,255,0))
        else:
            drawEllipse((0,0,255))

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release() 