import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import PIL
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


class draw_predict(tk.Tk):
	pil_image = PIL.Image.new("L", (100, 100))
	pil_draw = ImageDraw.Draw(pil_image)
	labelText = "Press and draw"

	def __init__(self):
		tk.Tk.__init__(self)
		self.canvas = tk.Canvas(self, width=100, height=100, bg = "black")
		self.canvas.pack()
		
		self.canvas.pack(side="top", fill="both", expand=True)
		self.canvas.bind("<B1-Motion>", self.draw)

		self.button=tk.Button(text="save",command=self.save)
		self.button.pack()

		self.button_predict = tk.Button(self, text = "Guess it!", command = self.make_prediction)
		self.button_predict.pack( fill="both", expand=True)

		self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
		self.button_clear.pack( fill="both", expand=True)

		self.message = tk.Label(self,text = self.labelText)
		self.message.pack(side = 'bottom')

	def draw(self,event):
		color = 'white'
		x1,y1 = (event.x - 1), (event.y - 1)
		x2,y2 = (event.x + 1), (event.y + 1)
		self.canvas.create_rectangle(x1,y1,x2,y2,fill = color, outline = color)
		self.pil_draw.rectangle([(x1,y1),(x2,y2)],fill= color)
	
	def save(self):
		print(np.asarray(self.pil_image.resize((28,28))))
		self.canvas.update()
		filename = "image.png"
		self.pil_image.resize((28,28)).save(filename)

	def clear_all(self):
		self.canvas.delete("all")
		self.pil_image = PIL.Image.new("L", (100, 100))
		self.pil_draw = ImageDraw.Draw(self.pil_image)

	def make_prediction(self):
		filename = 'softmax_model.sav'
		loaded_model = joblib.load(filename)
		a = np.asarray(self.pil_image.resize((28,28))).reshape(1,-1)
		print(a)
		predict = loaded_model.predict(a.reshape(1,-1))
		self.message['text'] = "It looks like number " + predict[0]


a = draw_predict()
a.mainloop()