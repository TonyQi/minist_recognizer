import tkinter as tk
from tkinter import Canvas,Button
import MINSTRecognizer
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt

def center_window(width, height):
    # 获取屏幕宽度和高度
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 计算X和Y偏移量
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    return f"{width}x{height}+{x}+{y}"

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字输入")

        self.canvas = Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(expand=False, fill='both')

        self.old_x = None
        self.old_y = None
        self.line_width = 2
        self.color = 'black'
        self.line_id = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.canvas.bind("<Button-3>", self.erase_last_line)

        self.recognize_button = Button(root, text="识别数字", command=self.recognize)
        self.recognize_button.pack(pady=10)

        self.label = tk.Label(root, text="")
        self.label.pack()

    def paint(self, event):
        if self.old_x and self.old_y:
            self.line_id = self.canvas.create_line((self.old_x, self.old_y, event.x, event.y),
                                                   width=20, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def erase_last_line(self, event):

        self.canvas.delete('all')
        self.line_id = None  # 重置线条ID，以便下次使用

    def recognize(self):
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()

        root_x = int(root_x / 0.5)
        root_y = int(root_y / 0.5)

        x = root_x + self.canvas.winfo_x()
        y = root_y + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()/0.5
        y1 = y + self.canvas.winfo_height()/0.5

        # 使用Pillow库截取画布区域的图像
        image = ImageGrab.grab().crop((x, y, x1, y1))

        image = image.resize((28, 28), Image.Resampling.BILINEAR)
        image_data = np.array(image.convert('L'),'f')
        # plt.imshow(image_data, cmap='gray')
        # plt.grid(False)
        #plt.show()
        recognizer = MINSTRecognizer.MINSTRecognizer()

        output = recognizer.recognize(image_data)
        self.label.config(text="识别结果为"+str(output))
        # print(output)

if __name__ == "__main__":

    root = tk.Tk()
    #root.geometry(center_window(400, 350))
    app = DrawingApp(root)
    root.mainloop()