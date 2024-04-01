from serial.tools import list_ports
import pydobot


available_ports = list_ports.comports()

port = available_ports[0].device

device = pydobot.Dobot(port=port, verbose=False)

(x_, y_, z_, r, j1, j2, j3, j4) = device.pose()

device.move_to(260, 0, -60, r, wait=True)  # we wait until this movement is done before continuing
