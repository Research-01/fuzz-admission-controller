from .config import ENABLE_PLOT, HEADLESS

if ENABLE_PLOT:
    import matplotlib
    if HEADLESS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self):
        if not ENABLE_PLOT:
            raise RuntimeError("Plotting is disabled; set ENABLE_PLOT=True to use LivePlot.")

        if not HEADLESS:
            plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, sharex=self.ax1)

        (self.l_fric,) = self.ax1.plot([], [], label="Friction (Mahalanobis)", color="red")
        self.ax1.set_ylabel("Distance")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper right")
        self.ax1.set_title("Mahalanobis Friction (Magnitude)")

        (self.l_eng,) = self.ax2.plot([], [], label="Energy (adaptive mean |ΔF|)", color="orange")
        self.ax2.set_ylabel("Energy")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right")
        self.ax2.set_title("System Energy (Instability)")

        self.fig.tight_layout()

    def update(self, t, fric, eng):
        n = len(t)
        if not (len(fric) == n and len(eng) == n):
            return
        self.l_fric.set_data(t, fric)
        self.l_eng.set_data(t, eng)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        if not HEADLESS:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def save(self, path: str):
        self.fig.savefig(path, dpi=140, bbox_inches="tight")
