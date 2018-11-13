import tkinter as tk
import lib
from PIL import ImageTk


class CameraFrame(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.size = (400, 400)

        self.transform = lib.Transform.identity()

        self.polyhedron = lib.Polyhedron.Cube(lib.Point(0, 0, 0), 0)
        self.camera = lib.Camera.iso()

        self.render = tk.Label(self)
        self.render.grid(row=0, column=0, rowspan=4)

        self.pos = [0, 0, 0]
        self.angles = [0, 0, 0]

        self.pos_vars = []
        self.pos_entries = []
        for i in range(3):
            self.pos_vars.append(tk.StringVar(self))
            self.pos_vars[i].trace('w', self.read_pos)
            self.pos_entries.append(tk.Entry(self, textvar=self.pos_vars[i]))
            tk.Label(self, text=chr(ord('x') + i)).grid(row=0, column=i*2+1)
            self.pos_entries[i].grid(row=0, column=(i + 1) * 2)

        self.key_bindid = root.bind('<Key>', self.keyboard_rotate)

        self.draw()

    def read_pos(self, *args):
        try:
            for i in range(3):
                self.pos[i] = int(self.pos_vars[i].get())
        except:
            pos = [0, 0, 0]
        self.camera = lib.Camera.persp(0.01, self.pos, self.angles)

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
        self.draw()

    def draw(self, *args):
        transformed = self.polyhedron.apply_transform(self.transform)
        self.im = self.camera.draw_with_both_culling_and_zbuf(self.size, transformed)
        # self.im.show()
        self.pim = ImageTk.PhotoImage(self.im)
        self.render.configure(image=self.pim)
