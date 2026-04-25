"""
Generate synthetic chart samples with known ground-truth data.
This allows us to validate extraction accuracy precisely.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

def save_meta(name, data_dict, axis_info):
    meta = {"data": data_dict, "axes": axis_info}
    with open(SAMPLES_DIR / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def sample_01_simple_linear():
    """Simple line chart with single x and y axis."""
    x = np.linspace(0, 10, 50)
    y = np.sin(x) * 10 + 20
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color="blue", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Simple Linear Chart")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "01_simple_linear.png")
    plt.close(fig)
    save_meta("01_simple_linear", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 10},
               "y": {"type": "linear", "min": 0, "max": 35}})

def sample_02_log_y():
    """Semi-log plot: y-axis is logarithmic."""
    x = np.linspace(1, 100, 100)
    y = np.exp(x / 20) + 1
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.semilogy(x, y, color="red", linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 200)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Semi-Log Y Axis")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "02_log_y.png")
    plt.close(fig)
    save_meta("02_log_y", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 100},
               "y": {"type": "log", "min": 1, "max": 200}})

def sample_03_loglog():
    """Log-log plot."""
    x = np.logspace(0, 3, 100)
    y = x ** 1.5
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.loglog(x, y, color="green", linewidth=2)
    ax.set_xlim(1, 1000)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel("X Value")
    ax.set_ylabel("Y Value")
    ax.set_title("Log-Log Chart")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "03_loglog.png")
    plt.close(fig)
    save_meta("03_loglog", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "log", "min": 1, "max": 1000},
               "y": {"type": "log", "min": 1, "max": 1e5}})

def sample_04_dual_y():
    """Dual y-axis chart."""
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) * 50 + 100
    y2 = np.cos(x) * 0.5 + 2
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    ax1.plot(x, y1, color="blue", linewidth=2, label="Temperature")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 160)
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Temperature (K)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="orange", linewidth=2, label="Pressure")
    ax2.set_ylim(0, 4)
    ax2.set_ylabel("Pressure (atm)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax1.set_title("Dual Y-Axis Chart")
    fig.savefig(SAMPLES_DIR / "04_dual_y.png")
    plt.close(fig)
    save_meta("04_dual_y",
              {"series1": {"x": x.tolist(), "y": y1.tolist()},
               "series2": {"x": x.tolist(), "y": y2.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 10},
               "y_left": {"type": "linear", "min": 0, "max": 160},
               "y_right": {"type": "linear", "min": 0, "max": 4}})

def sample_05_inverted_axis():
    """Chart with inverted y-axis."""
    x = np.linspace(0, 10, 50)
    y = x ** 2
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color="purple", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(120, 0)  # inverted
    ax.set_xlabel("X")
    ax.set_ylabel("Y (inverted)")
    ax.set_title("Inverted Y Axis")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "05_inverted_y.png")
    plt.close(fig)
    save_meta("05_inverted_y", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 10},
               "y": {"type": "linear", "min": 0, "max": 120, "inverted": True}})

def sample_06_scatter():
    """Scatter plot."""
    np.random.seed(42)
    x = np.random.uniform(0, 100, 30)
    y = np.random.uniform(0, 50, 30)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter(x, y, color="teal", s=60, edgecolors="black")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_xlabel("X Variable")
    ax.set_ylabel("Y Variable")
    ax.set_title("Scatter Plot")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "06_scatter.png")
    plt.close(fig)
    save_meta("06_scatter", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 100},
               "y": {"type": "linear", "min": 0, "max": 60}})

def sample_07_multi_series():
    """Multiple line series."""
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) * 10
    y2 = np.cos(x) * 8
    y3 = np.sin(x) * np.cos(x) * 12
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y1, color="blue", linewidth=2, label="A")
    ax.plot(x, y2, color="red", linewidth=2, label="B")
    ax.plot(x, y3, color="green", linewidth=2, label="C")
    ax.set_xlim(0, 10)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Series Line Chart")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "07_multi_series.png")
    plt.close(fig)
    save_meta("07_multi_series",
              {"series1": {"x": x.tolist(), "y": y1.tolist()},
               "series2": {"x": x.tolist(), "y": y2.tolist()},
               "series3": {"x": x.tolist(), "y": y3.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 10},
               "y": {"type": "linear", "min": -15, "max": 15}})

def sample_08_log_x():
    """Semi-log plot: x-axis is logarithmic."""
    x = np.logspace(0, 4, 100)
    y = np.sqrt(x)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.semilogx(x, y, color="brown", linewidth=2)
    ax.set_xlim(1, 10000)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Semi-Log X Axis")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "08_log_x.png")
    plt.close(fig)
    save_meta("08_log_x", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "log", "min": 1, "max": 10000},
               "y": {"type": "linear", "min": 0, "max": 120}})

def sample_09_no_grid():
    """Simple chart without grid lines."""
    x = np.linspace(0, 5, 30)
    y = np.exp(-x) * 20
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color="darkgreen", linewidth=2, marker="o", markersize=4)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Time")
    ax.set_ylabel("Decay")
    ax.set_title("No Grid Lines")
    fig.savefig(SAMPLES_DIR / "09_no_grid.png")
    plt.close(fig)
    save_meta("09_no_grid", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 5},
               "y": {"type": "linear", "min": 0, "max": 25}})

def sample_10_dense_data():
    """Dense line data."""
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(3 * x) + 0.5 * np.cos(7 * x)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color="navy", linewidth=1)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Value")
    ax.set_title("Dense Data")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(SAMPLES_DIR / "10_dense.png")
    plt.close(fig)
    save_meta("10_dense", {"series1": {"x": x.tolist(), "y": y.tolist()}},
              {"x": {"type": "linear", "min": 0, "max": 2 * np.pi},
               "y": {"type": "linear", "min": -2, "max": 2}})

if __name__ == "__main__":
    sample_01_simple_linear()
    sample_02_log_y()
    sample_03_loglog()
    sample_04_dual_y()
    sample_05_inverted_axis()
    sample_06_scatter()
    sample_07_multi_series()
    sample_08_log_x()
    sample_09_no_grid()
    sample_10_dense_data()
    print(f"Generated 10 sample charts in {SAMPLES_DIR}")
