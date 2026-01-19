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
        self.fig = plt.figure(figsize=(14, 10))
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        (self.l_fric,) = self.ax1.plot([], [], label="Friction (Magnitude)", color="red")
        self.ax1.set_ylabel("Distance")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper right")
        self.ax1.set_title("Mahalanobis Friction (Magnitude)")

        self.ax2 = self.fig.add_subplot(3, 1, 2, sharex=self.ax1)
        (self.l_eng,) = self.ax2.plot([], [], label="Energy (adaptive mean |ΔF|)", color="orange")
        self.ax2.set_ylabel("Energy")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right")
        self.ax2.set_title("System Energy (Instability)")

        self.ax3 = self.fig.add_subplot(3, 1, 3, sharex=self.ax1)
        (self.l_dir,) = self.ax3.plot([], [], label="Direction (+1/-1)", color="blue", drawstyle="steps-post")
        self.ax3.axhline(0, color="black", linewidth=1, linestyle="--")
        self.ax3.set_ylabel("Direction")
        self.ax3.set_ylim(-1.5, 1.5)
        self.ax3.set_yticks([-1, 0, 1])
        self.ax3.set_yticklabels(["Below", "0", "Above"])
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc="upper right")
        self.ax3.set_title("Direction (+1: Stress, -1: Headroom)")

        self.fig.tight_layout()

    def update(self, t, fric, eng, direc):
        n = len(t)
        if not (len(fric) == n and len(eng) == n and len(direc) == n):
            return
        self.l_fric.set_data(t, fric)
        self.l_eng.set_data(t, eng)
        self.l_dir.set_data(t, direc)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        if not HEADLESS:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def save(self, path: str):
        self.fig.savefig(path, dpi=140, bbox_inches="tight")
