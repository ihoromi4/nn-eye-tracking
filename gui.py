import tkinter as tk
import numpy as np


class Circle:
    def __init__(self, canvas, pos=(0, 0), radius=1, fill="#000000"):
        self.canvas = canvas

        self.id = canvas.create_oval(0, 0, 0, 0, fill=fill, width=0.0)

        self.radius = radius
        self._pos = np.array(pos)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = pos
        coords = (
            pos[0] - self._radius,
            pos[1] - self._radius,
            pos[0] + self._radius,
            pos[1] + self._radius
        )
        coords = list(map(int, coords))
        self.canvas.coords(self.id, coords)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def fill(self):
        pass

    @fill.setter
    def fill(self, fill):
        self.canvas.itemconfig(self.id, fill=fill)

    def delete(self):
        self.canvas.delete(self.id)


class Window(tk.Tk):
    def __init__(self):
        super().__init__()

        self.canvas = tk.Canvas(self, bg="white")
        self.dt = 30

        self.attributes('-zoomed', True)

    def create_circle(self, *args, **kwargs):
        return Circle(self.canvas, *args, **kwargs)

    def get_size(self):
        return np.array([self.canvas.winfo_width(), self.canvas.winfo_height()])

    def _handle_idle(self):
        self.on_update()

        self.after(self.dt, self._handle_idle)

    def on_update(self):
        pass

    def start(self):
        self.after(0, self._handle_idle)
        self.canvas.pack(fill="both", expand=True)
        self.mainloop()

