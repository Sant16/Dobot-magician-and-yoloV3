import cv2
import numpy as np
from serial.tools import list_ports
import pydobot
import base64

def take_pic():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None, None  # Consider how you want to handle errors
    altura, anchura = frame.shape[:2]
    print("Anchura:", anchura)
    print("Altura:", altura)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    return blob, frame  # Returning both the blob and the frame

def obj_detc(blob, frame, net, layer_names, output_layers):
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    boxes = []
    confidences = []
    x_cor = []
    y_cor = []
    h_ = []
    w_ = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > 0.85:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            w_.append(boxes[i][2])
            h_.append(boxes[i][3])
            x_cor.append(boxes[i][0])
            y_cor.append(boxes[i][1])
            print(x_cor)

    return x_cor, y_cor, w_, h_  # Returning the coordinates


def marcar_y_mostrar(frame, x_cor, y_cor, w_, h_):
    """
    Dibuja marcas en los objetos detectados y muestra la imagen resultante.

    Parámetros:
    - frame: La imagen original en la que se detectaron los objetos.
    - x_cor: Lista de las coordenadas x de los objetos detectados.
    - y_cor: Lista de las coordenadas y de los objetos detectados.
    - w_: Lista de los anchos de los objetos detectados.
    - h_: Lista de las alturas de los objetos detectados.
    """
    # Dibuja un círculo en el centro de cada objeto detectado
    for r in range(len(x_cor)):
        centro_x = int(x_cor[r] + w_[r]/2)
        centro_y = int(y_cor[r] + h_[r]/2)
        cv2.circle(frame, (centro_x, centro_y), 5, (0,255,255), -1)
        cv2.imwrite('/home/srm/Documents/dobot_/dobot_API/static/detection.jpg',frame)
    # # Muestra la imagen procesada
    # cv2.imshow("Objetos Marcados", frame)
    # cv2.waitKey(0)  # Espera hasta que una tecla sea presionada para cerrar la ventana
    # cv2.destroyAllWindows()  # Cierra las ventanas abiertas




def dobot_coor(frame, x_cor, y_cor, w_, h_):
    
    distancia_efector = 19
    px_cubos = 520 #px beacuse x in image is in horizontal different from dobot
    cm_cubos = 30
    pix_efector_final = (distancia_efector * px_cubos)/cm_cubos
    obx = []
    oby = []
    for r in range(len(x_cor)):
        # x,y in image =! x,y in dobot
        y_ = (x_cor[r] + w_[r]/2) 
        x_ = (y_cor[r] + h_[r]/2)

        py = y_ - pix_efector_final                             
        y_ = (py * cm_cubos)/px_cubos                                 
        y_ = int(y_*10) 
        x_ = int(((x_ * (cm_cubos*10))/px_cubos) + 105)  
        obx.append(x_)
        oby.append(y_)
        print(f'{obx}/{oby}')

    return obx, oby

def dobot_mov(obx, oby, device):
    offset = -35 #290 
    for i in range(len(obx)):

        x_ = obx[i]
        y_ = oby[i]

        device.move_to(x_, y_, 0, 0, wait=False)  # we wait unt
        device.move_to(x_, y_, -44, 90, wait=True)  # we wait unt
        device.grip(True)
        device.move_to(x_, y_, 60, 90, wait=True)
        device.move_to(200, 0, 60, 90, wait=True)
        device.move_to(200, 0, offset, 90, wait=True)
        device.grip(False)
        # device.move_to(x_, y_, 10, 0, wait=False)  # we wait unt
        # device.move_to(offset, 0, 10, 90, wait=True)  # we wait unt
        # device.move_to(offset, 0, -39, 0, wait=True)  # we wait unt
        # device.grip(False)
        # device.move_to(offset, 0, 10, 90, wait=True)  # we wait unt
        offset = offset + 32

def start_process():
    available_ports = list_ports.comports()
    port = available_ports[0].device
    device = pydobot.Dobot(port=port, verbose=False)
   
    device.speed(2500,2500)
    device.move_to(100, 250, 0, 90, wait=True)  # we wait until this movement is done before continuing
    device.speed(2100,2100)
    device.grip(False)

    # Cargar YOLO
    net= cv2.dnn.readNetFromDarknet('/home/srm/Documents/dobot_/darknet/cfg/yolov3_cubes_test.cfg',
                                            '/home/srm/Documents/dobot_/darknet/backup/yolov3_cubes_train_final.weights')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    blob, frame = take_pic()
    if blob is not None:
        # Llama a obj_detc para obtener las coordenadas de los objetos detectados
        x_cor, y_cor, w_, h_ = obj_detc(blob, frame, net, layer_names, output_layers)  # Desempaquetando las coordenadas directamente

        # Llama a marcar_y_mostrar para dibujar marcas en los objetos detectados y mostrar el resultado
        marcar_y_mostrar(frame, x_cor, y_cor, w_, h_)
        x_list, y_list = dobot_coor(frame, x_cor, y_cor, w_, h_)
        dobot_mov(x_list, y_list, device)



if __name__ == "__main__":
    start_process()



