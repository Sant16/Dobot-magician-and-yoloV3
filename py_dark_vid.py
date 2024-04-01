import cv2
import numpy as np
import time
from serial.tools import list_ports
import pydobot


available_ports = list_ports.comports()

port = available_ports[0].device

device = pydobot.Dobot(port=port, verbose=False)

(x_, y_, z_, r, j1, j2, j3, j4) = device.pose()

device.move_to(300, 0, -60, r, wait=True)  # we wait until this movement is done before continuing



# Cargar YOLO
net= cv2.dnn.readNetFromDarknet('/home/srm/Documents/dobot_/darknet/cfg/yolov3_cubes_test.cfg',
                                         '/home/srm/Documents/dobot_/darknet/backup/yolov3_cubes_train_final.weights')
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Inicializar la captura de video
cap = cv2.VideoCapture(1)

ret, frame = cap.read()
altura, anchura = frame.shape[:2]
print("Anchura:", anchura)
print("Altura:", altura)
flag = False
flag_time = False
while True:

    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            # Extraer la confianza y las coordenadas del cuadro delimitador
            scores = detection[5:]
            confidence = scores[0]  # Suponiendo que 'cube' es la única clase y está en la primera posición

            if confidence > 0.85:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    cv2.line(frame, (250, 0), (250, 400), (255, 0, 0), 2)
    cv2.line(frame, (390, 0), (390, 400), (255, 0, 0), 2)
    
    if len(idxs) > 0:
        
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #print(boxes)
            #cv2.rectangle(frame, (x , y), (x + w, y + h), (0, 255, 0), 2)  # Color verde para los cuadros
            cv2.circle(frame, (int(x + w/2), int(y + h/2)), 5, (0,255,255),-1)
            # text = "Cube: {:.4f}".format(confidences[i])
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            obj = (x + w/2) #coordenada X
            obj_y = (y + h/2)
            distancia_efector = 19 
            px_cubos = 520 ## between cubes
            cm_cubos = 30  ## between cubes
            pix_efector_final = (distancia_efector * px_cubos)/cm_cubos
            px = obj - pix_efector_final #362 es la distancia del efector final en pixeles.
                            #px es la distancia del objeto en pixeles con respecto al efector final
            y_ = (px * cm_cubos)/px_cubos #18 es la distancia en cm calibrada con cubos y 362 pixeles calibrados con cubos 
                                #coincidencialemte 362 dio igual arriba pero no necesariamente debe ser asi
            y_ = int(y_*10) # dr es la distancia en milimetros del objeto con respecto al efector final
            x_ = int(((obj_y * (cm_cubos*10))/px_cubos) + 110)  
            #180 son los 18 cm calibrados con los cubos, obj_y los pixeles en 'y', 362 pixeles calibrados, 170 la distancia del
            #efector final a 0 pixeles en 'y' desde su inici

            #print(f'{y_}/{x_}')
            print(y_)
            device.move_to(x_, y_, -30, r, wait=False)  # we wait until this movement is done before continuing

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()