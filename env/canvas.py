import tkinter as tk

window = tk.Tk()
window.title('my window')


x0, y0, x1, y1 = 50, 50, 80, 80
x2, y2, x3, y3 = 80, 80, 100, 100
box_w = 10   # number of units in a row
box_h = 6   # number of units in a column
units = 40  # size of unit

window.geometry('{0}x{1}'.format(box_w * units, box_h * units))


canvas = tk.Canvas(window, bg='white', height=box_h * units, width=box_w * units)


rec = canvas.create_rectangle(x0, y0, x1, y1, fill='red')
rec2 = canvas.create_rectangle(x2, y2, x3, y3, fill='black')

for c in range(0, box_w * units, units):
    x0, y0, x1, y1 = c, 0, c, box_h * units
    v_line = canvas.create_line(x0, y0, x1, y1)

for r in range(0, box_h * units, units):
    x0, y0, x1, y1 = 0, r, box_w * units, r
    h_line = canvas.create_line(x0, y0, x1, y1)

canvas.pack()


def moveit():

    canvas.move(rec, 0, -2)
    canvas.move(rec2, 2, 0)


b = tk.Button(window, text='move', command=moveit).pack()

window.mainloop()