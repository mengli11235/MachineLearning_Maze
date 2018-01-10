import tkinter as tk

import numpy as np





window = tk.Tk()

window.title('my window')





x0, y0, x1, y1 = 50, 50, 80, 80

x2, y2, x3, y3 = 80, 80, 100, 100

box_w = 10   # number of units in a row

box_h = 10   # number of units in a column

size = 40  # size of unit



window.geometry('{0}x{1}'.format(box_w * size, box_h * size))





canvas = tk.Canvas(window, bg='white', height=box_h * size, width=box_w * size)





for c in range(0, box_w * size, size):

    x0, y0, x1, y1 = c, 0, c, box_h * size

    v_line = canvas.create_line(x0, y0, x1, y1)



for r in range(0, box_h * size, size):

    x0, y0, x1, y1 = 0, r, box_w * size, r

    h_line = canvas.create_line(x0, y0, x1, y1)





def obtacle(x_position, y_position):

    x_p = x_position -1

    y_p = y_position -1

    origin = np.array([20, 20])

    print(origin)

    ob_center = origin + np.array([size * x_p, size * y_p])

    obtacles = canvas.create_rectangle(

                ob_center[0] - 15, ob_center[1] - 15,

                ob_center[0] + 15, ob_center[1] + 15,

                fill='black')

    return obtacles





def reword(x_position, y_position):

    x_p = x_position -1

    y_p = y_position -1

    origin = np.array([20, 20])

    print(origin)

    ob_center = origin + np.array([size * x_p, size * y_p])

    rewords = canvas.create_oval(

                ob_center[0] - 15, ob_center[1] - 15,

                ob_center[0] + 15, ob_center[1] + 15,

                fill='red')

    return rewords





obtacle(3, 6)

obtacle(3, 7)

obtacle(3, 8)

obtacle(4, 6)

obtacle(5, 6)

obtacle(6, 6)

obtacle(8, 8)

obtacle(9, 6)

obtacle(6, 7)

obtacle(7, 8)

obtacle(8, 6)

obtacle(8, 7)

obtacle(6, 8)



reword(5, 7)





canvas.pack()





window.mainloop()