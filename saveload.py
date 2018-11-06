import lib
import tkinter as tk
import tkinter.filedialog

class SaveLoad(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.open = tk.Button(self, text="open", command=self.open_file).grid(row=1, column=0)
        self.save = tk.Button(self, text="save", command=self.save_file).grid(row=1, column=1)

    def open_file(self):
        filename = tk.filedialog.askopenfilename()
        self.polyhedron = lib.Polyhedron.load_obj(filename)

    def save_file(self):
        filename = tk.filedialog.asksaveasfilename()
        self.polyhedron.save_obj(filename)
        
    def draw(self):
        pass
