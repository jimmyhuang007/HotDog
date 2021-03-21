# --------------------------From TensorFLow Example-----------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import picamera
import argparse
import numpy as np

from io import BytesIO
from PIL import ImageTk, Image
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

# --------------------------From TensorFLow Example-----------------------------


# Root initial
root = tk.Tk()

# Camera initial
cameraWidth = 640
cameraHeight = 480
camera = picamera.PiCamera(resolution=(cameraWidth, cameraHeight), framerate=30)

class Process:
  def __init__(self, val):
    self.__state = val
    self.__interpreter = None
    self.__height = 0
    self.__width = 0
    self.__labels = None

  # State 0 is inital state, preivew only
  # State 1 is processing state, no preview

  def get_state(self):
    return self.__state

  def set_state(self, val):
    self.__state = val

  def get_interpreter(self):
    return self.__interpreter

  def set_interpreter(self, obj):
    self.__interpreter = obj

  def get_height(self):
    return self.__height

  def set_height(self, val):
    self.__height = val

  def get_width(self):
    return self.__width

  def set_width(self, val):
    self.__width = val

  def get_labels(self):
    return self.__labels

  def set_labels(self, obj):
    self.__labels = obj


class UI:
  def __init__(self):
    # Top Frame initial
    self.__topFrame = tk.Frame(root, bg='black')
    self.__topFrame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Bottom Frame initial
    self.__botFrame = tk.Frame(root, bg='black')
    self.__botFrame.pack(side=tk.BOTTOM)

    # Snap Button Inital
    self.__snapButton = tk.Button(self.__botFrame, text="Snap", command=myButton, bg='#cd127e', height=1, width=7)
    self.__snapButton.config(fg='white', font=("Segoe Script", 14, 'bold'))
    self.__snapButton.pack(pady=10)

    # resultLabel initial
    self.__resultLabel = tk.Label(root, text="Nothing", bg='black')
    self.__resultLabel.config(fg='black', font=("Segoe Script", 24, 'bold'))
    self.__resultLabel.pack(pady=10, side=tk.BOTTOM)

    # img initial
    self.__imgLabel = tk.Label(root, image="", bg='black')
    self.__imgLabel.place(x=0, y=20)

  def get_snapButton(self):
      return self.__snapButton

  def get_topFrame(self):
      return self.__topFrame

  def get_botFrame(self):
      return self.__botFrame

  def get_resultLabel(self):
      return self.__resultLabel

  def get_imgLabel(self):
      return self.__imgLabel


# Button function
def myButton():

  curProcess = myProcess
  curUI = myUI

  state = curProcess.get_state()
  labels = curProcess.get_labels()

  snapButton = curUI.get_snapButton()
  resultLabel = curUI.get_resultLabel()
  imgLabel = curUI.get_imgLabel()
  botFrame = curUI.get_botFrame()

  if state == 0:
    curProcess.set_state(1)

    # Change Button
    snapButton['text'] = "Another One?"
    snapButton['width'] = 10
    botFrame.pack(anchor="s", side=tk.RIGHT, padx=10)

    # Capture Image
    stream = BytesIO()
    camera.capture(stream, 'jpeg')
    camera.stop_preview()
    
    # Test from local upload
    # stream.seek(0)
    # myImg = Image.open("hotdog1.jpeg")
    # myImg.save(stream, format='png')

    # Store Image
    stream.seek(0)
    displayImage = Image.open(stream).resize((cameraWidth, cameraHeight),Image.ANTIALIAS)
    stream.seek(0)
    inputImage = Image.open(stream).convert('RGB').resize((curProcess.get_height(), curProcess.get_width()), Image.ANTIALIAS)

    # Display Image
    tkImg = ImageTk.PhotoImage(displayImage)
    imgLabel["image"] = tkImg # Calling it twice works... 
    imgLabel.image = tkImg
    imgLabel.config(image = tkImg)
    

    # Classify Imag
    results = classify_image(curProcess.get_interpreter(), inputImage)
    lid, prob = results[0]
    
    objName = labels[lid]
    print("Item:", objName, "Prob:", prob)

    if objName == "hotdog":
      resultLabel['text'] = "HotDog"
      resultLabel['fg'] = '#0bfc03'
    else:
      resultLabel['text'] = "Not HotDog"
      resultLabel['fg'] = '#fc7303'

  else:
      # Back to org
      curProcess.set_state(0)
      imgLabel.config(image = "")
      root['bg'] = 'black'
      snapButton['text'] = "Snap"
      snapButton['width'] = 7
      botFrame.pack(side=tk.BOTTOM)
      resultLabel['fg'] = 'black'
      preview()

# Preview Function update the location of the preview window
def preview():
    camera.start_preview(alpha=255, fullscreen=False, window=(root.winfo_x(), root.winfo_y() + 20, 640, 480))


# Dragging action
def Drag(event):
  curProcess = myProcess
  if ((event.widget) is root) and (not curProcess.get_state() == 1):
    preview()  # Update the preview window to reflect change in x-y coordinates
root.bind('<Configure>', Drag) # binds Drag to configure which is called whenever window move


# Customized Root Window
root.geometry("640x560")  # model accepts 640x480 img, bit bigger for padding
root.configure(bg='black')
root.resizable(False, False)
root.attributes('-topmost', True)

# Create UI and Process
myUI = UI()
myProcess = Process(0)

# onStart of tkinter window
def onStart():
    print("App Start")
    preview()

# onClose of tkinter window
def onClose():
    print("App Close")
    camera.stop_preview()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", onClose)


# Parse argument line for loading image recognition components
def main():

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='File path of .tflite file.', required=True)
  parser.add_argument('--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  # Modifiy myProcess
  myProcess.set_interpreter(interpreter)
  myProcess.set_height(height)
  myProcess.set_width(width)
  myProcess.set_labels(labels)

  # Execute tkinter UI
  root.wait_visibility()  # Wait for start
  onStart()
  root.mainloop()


if __name__ == '__main__':
  main()

# python3 HotDog.py --model inception_v4_299_quant.tflite --labels labels.txt
