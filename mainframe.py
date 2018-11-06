import lib
import tkinter as tk
from lab6 import Lab6
from graphics import Graphic
from saveload import SaveLoad
from screw import Screw


class Main:
    def __init__(self, root):
        self.root = root

        self.ui_list = [
            'lab6',
            'graphic',
            'save-load',
            'screw'
        ]

        self.polyhedron = lib.Polyhedron.Cube(lib.Point(0, 0, 0), 100)
        self.camera = lib.Camera.ortho()
        self.transform = lib.Transform.identity()

        self.current = 0

        self.menu_var = tk.StringVar()
        self.menu_var.set(self.ui_list[self.current])
        self.menu = tk.OptionMenu(root, self.menu_var, *self.ui_list)
        self.menu.grid(row=0, column=0)

        self.frames = [Lab6(root),
                       Graphic(root),
                       SaveLoad(root),
                       Screw(root)]
        self.frames[self.current].grid(row=1, column=0)

        self.menu_var.trace("w", self.change_menu)

    def change_menu(self, *args):
        tr = self.menu_var.get()
        idx = self.ui_list.index(tr)
        self.frames[self.current].grid_forget()
        self.polyhedron = self.frames[self.current].polyhedron
        self.frames[idx].polyhedron = self.polyhedron
        fr = self.frames[idx]
        fr.grid(row=1, column=0)
        self.current = idx
        self.frames[self.current].draw()
