import cv2
import numpy as np


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

            if confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    cv2.line(frame, (250, 0), (250, 400), (255, 0, 0), 2)
    cv2.line(frame, (390, 0), (390, 400), (255, 0, 0), 2)
    cv2.line(frame, (40, 220), (600, 220), (255, 255, 0), 2)
    
    if len(boxes) == 2:
            
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                obj1_x = boxes[0][0] + (boxes[0][2]/2)
                obj2_x = boxes[1][0] + (boxes[1][2]/2)

                obj1_y = boxes[0][1] + (boxes[0][2]/2)
                obj2_y = boxes[1][1] + (boxes[1][2]/2)

                print(f'ob1 = {obj1_x}')
                print('//////')
                print(f'ob2 = {obj2_x}')
                print('//////')

                cv2.rectangle(frame, (x , y), (x + w, y + h), (0, 255, 0), 2)  # Color verde para los cuadros
                cv2.circle(frame, (int(x + w/2), int(y + h/2)), 5, (0,255,255),-1)
                # text = "Cube: {:.4f}".format(confidences[i])
                # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                px = abs(obj2_y - obj1_y)
                
                #print(f'px = {px}')
               
                
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()