import tkinter as tk
import lib


class Screw(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.width = 300
        self.height = 200

        self.polyhedron = lib.Polyhedron.Cube(lib.Point(0, 0, 0), 0)
        self.camera = lib.Camera.iso()

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='white')
        self.canvas.grid(row=0, column=0)

        self.canvas.bind("<Button-1>", self.make_point)
        self.points = []

        self.axis_var = tk.StringVar()
        self.axis = tk.OptionMenu(self, self.axis_var, 'x', 'y', 'z')
        self.axis.grid(row=1, column=0)
        self.axis_var.set("x")
        self.axis_var.trace('w', self.read_params)

        self.angle = 0.
        self.separation_var = tk.StringVar()
        self.separation = tk.Entry(self, textvar=self.separation_var)
        self.separation.grid(row=2, column=0)
        self.separation_var.trace('w', self.read_params)

        self.clear_button = tk.Button(self, text="clear", command=self.clear)
        self.clear_button.grid(row=3, column=0)

        self.render = tk.Label(self, width=self.width, height=self.height)
        self.render.grid(row=4, column=0)

        self.canvas_draw()

    def clear(self):
        self.points = []
        self.canvas_draw()

    def make_point(self, event):
        if (event.x >= 0) and (event.x < self.width) and (event.y >= 0) and (event.y < self.height):
            self.points.append((event.x, event.y))
            self.canvas_draw()

    def read_params(self, *args):
        try:
            self.angle = 360. / float(self.separation_var.get())
        except:
            self.angle = 0.
        self.screw()

    def canvas_draw(self):
        self.canvas.delete(tk.ALL)
        self.canvas.create_line(self.width / 2, 0, self.width / 2, self.height, fill='red')
        self.canvas.create_line(0, self.height / 2, self.width, self.height / 2, fill='red')
        for i in range(len(self.points) - 1):
            self.canvas.create_line(self.points[i], self.points[i + 1])

        for p in self.points:
            self.canvas.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1)

    def screw(self):
        points = [lib.Point(x, 0, z) for x, z in self.points]
        transform = lib.Transform.rotate(self.axis_var.get(), self.angle)
        polyhedron = lib.Polyhedron(points, [])
        sides = []
        print(self.angle)
        for _ in range(int(360. / self.angle)):
            polyhedron2 = polyhedron.apply_transform(transform)
            ind_now = len(points)
            ind_prev = ind_now - len(self.points)
            points += [lib.Point(x, y, z) for x, y, z in polyhedron2.points.T]

            for i in range(len(self.points) - 1):
                p1 = ind_prev + i
                p2 = ind_prev + i + 1
                q1 = ind_now + i
                q2 = ind_now + i + 1

                sides.append([p1, p2, q1])
                sides.append([p1, q1, q2])

            polyhedron = polyhedron2

        print(points)
        print(sides)
        self.polyhedron = lib.Polyhedron(points, sides)
        self.draw()

    def draw(self):
        im = self.camera.draw((self.width, self.height), self.polyhedron.points, self.polyhedron.sides)
        self.pim = ImageTk.PhotoImage(im)
        self.render.configure(image=self.pim)
