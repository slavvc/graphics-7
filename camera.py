import tkinter as tk
import lib
from PIL import ImageTk


class CameraFrame(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.size = (600, 800)

        self.transform = lib.Transform.identity()

        self.polyhedron = lib.Polyhedron.Cube(lib.Point(0, 0, 0), 0)

        self.render = tk.Label(self)
        self.render.grid(row=0, column=0, rowspan=4)

        self.pos = [0, 0, 0]
        self.angles = [0, 0, 0]
        self.camera = lib.Camera.persp(0.01, self.pos, self.angles)

        self.draw()

    def keyboard_rotate(self, e):
        if e.char == 'a':
            self.angles[0] += 0.1
        elif e.char == 'd':
            self.angles[0] -= 0.1
        elif e.char == 'w':
            self.angles[1] += 0.1
        elif e.char == 's':
            self.angles[1] -= 0.1
        elif e.char == 'q':
            self.angles[2] += 0.1
        elif e.char == 'e':
            self.angles[2] -= 0.1
        elif e.char == 'i':
            self.pos[1] += 5
        elif e.char == 'k':
            self.pos[1] -= 5
        elif e.char == 'j':
            self.pos[0] -= 5
        elif e.char == 'l':
            self.pos[0] += 5
        elif e.char == 'u':
            self.pos[2] += 5
        elif e.char == 'o':
            self.pos[2] -= 5

        self.camera = lib.Camera.persp(0.01, self.pos, self.angles)
        self.draw()

    def draw(self, *args):
        transformed = self.polyhedron.apply_transform(self.transform)
        self.im = self.camera.draw_with_both_culling_and_zbuf(self.size, transformed)
        # self.im.show()
        self.pim = ImageTk.PhotoImage(self.im)
        self.render.configure(image=self.pim)
