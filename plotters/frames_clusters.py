import cv2
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class AnnotatedCluster:
    def __init__(self, x, y, frames, frames_ids=None, labels=None, zoom=1, dpi=120):
        self.x = x
        self.y = y
        self.frames = frames
        self.frames_ids = frames_ids
        assert x.shape == y.shape, 'x and y must have the same shape'
        assert len(x) == frames.shape[0], 'num of frames must equal len of x and y'
        assert frames_ids is None or frames_ids.shape[0] == frames.shape[0], 'frames_ids must have same len as frames'

        self.xybox = (200., 200.)  # arrow distance
        self.fig, self.ax = plt.subplots(dpi=dpi)
        self.sc = self.ax.scatter(x, y, c=labels)
        self.im = OffsetImage(frames[0], zoom=zoom)
        self.ab = AnnotationBbox(self.im, (0, 0), xybox=self.xybox, xycoords='data', boxcoords="offset points",
                                 pad=0.3, arrowprops=dict(arrowstyle="->"))
        self.ax.add_artist(self.ab)
        self.ab.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.load_tk()

    def load_tk(self):
        window = tk.Tk()
        # setting the title
        window.title('Annotated Cluster')
        # window.geometry("500x500")

        def on_closing():
            plt.close(self.fig)
            window.destroy()

        window.protocol('WM_DELETE_WINDOW', on_closing)
        # creating the Tkinter canvas containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(self.fig, master=window)
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        # run the gui
        window.mainloop()

    def update_annot_image(self, ind, event):
        # get the figure size
        w, h = self.fig.get_size_inches() * self.fig.dpi
        ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
        hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        self.ab.xybox = (self.xybox[0] * ws, self.xybox[1] * hs)
        # # make annotation box visible
        self.ab.set_visible(True)
        # place it at the position of the hovered scatter point
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.ab.xy = pos
        # set the image corresponding to that point
        i = ind["ind"][0]
        frame = self.frames[i]
        if self.frames_ids is not None:
            frame = cv2.putText(frame, str(self.frames_ids[i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
        self.im.set_data(frame)

    def hover(self, event):
        vis = self.ab.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                # update_annot(ind)
                self.update_annot_image(ind, event)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.ab.set_visible(False)
                    self.fig.canvas.draw_idle()
