import argparse
import socket, struct
import numpy as np
from array import array

class Server:
	def __init__(self, port=8000, image_size=(200,66)):
		print('Started server')
		self.image_size = image_size
		self.buffer_size = image_size[0]*image_size[1]*3;
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.bind(('0.0.0.0', port))
		self.s.listen(1)

		self.conn, self.addr = self.s.accept()
		print('GTAV connected')

	def recvImage(self):
		data = b""
		while len(data) < self.buffer_size:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		return np.resize(np.fromstring(data, dtype='uint8'), (self.image_size[1], self.image_size[0], 3)).astype('float32')

	def sendCommands(self, throttle, steering):		
		data = array('f', [throttle, steering])
		self.conn.sendall(data.tobytes())
		print('Sent commands', data)

	def recvReward(self):
		data = b""
		while len(data) < 4:
			packet = self.conn.recv(self.buffer_size - len(data))
			if not packet: return None
			data += packet

		print('Received reward')
		return struct.unpack('f', data)[0]

	def close(self):
		self.s.close()