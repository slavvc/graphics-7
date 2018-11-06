import tkinter as tk

class Screw(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.canvas_width = 500
        self.canvas_height = 400

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.grid(row=0, column=0)

        self.canvas.bind("<Button-1>", self.make_point)
        self.points = []

        self.axis_var = tk.StringVar()
        self.axis = tk.OptionMenu(self, self.axis_var, 'OX', 'OY', 'OZ')
        self.axis.grid(row=0, column=1)
        self.axis_var.set("OX")
        self.axis_var.trace('w', self.change_axis)

        self.clear_button = tk.Button(self, text="clear", command=self.clear)
        self.clear_button.grid(row=0, column=2)

        self.canvas_draw()

    def clear(self):
        self.points = []
        self.canvas_draw()

    def make_point(self, event):
        if (event.x >= 0) and (event.x < self.canvas_width) and (event.y >= 0) and (event.y < self.canvas_height):
            self.points.append((event.x, event.y))
            self.canvas_draw()

    def change_axis(self, *args):
        pass

    def canvas_draw(self):
        self.canvas.delete(tk.ALL)
        self.canvas.create_line(self.canvas_width / 2, 0, self.canvas_width / 2, self.canvas_height, fill='red')
        self.canvas.create_line(0, self.canvas_height / 2, self.canvas_width, self.canvas_height / 2, fill='red')
        for i in range(len(self.points) - 1):
            self.canvas.create_line(self.points[i], self.points[i + 1])

        for p in self.points:
            self.canvas.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1)
