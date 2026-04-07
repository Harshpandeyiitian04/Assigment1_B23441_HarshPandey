"""
plot_results.py – Publication-quality figures for DEM Assignment 1
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FONTSIZE = 11
plt.rcParams.update({
    "font.family": "serif", "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE, "axes.titlesize": FONTSIZE,
    "legend.fontsize": FONTSIZE-1, "xtick.labelsize": FONTSIZE-1,
    "ytick.labelsize": FONTSIZE-1, "lines.linewidth": 1.6,
    "lines.markersize": 5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})
os.makedirs("figures", exist_ok=True)
g = 9.81

def plot_freefall():
    df = pd.read_csv("test1_freefall.csv")
    t, z_num = df["t"].values, df["z0"].values
    z0, R = 8.0, 0.5
    t_hit = np.sqrt(2*(z0-R)/g)
    mask = t <= t_hit
    t_a  = t[mask]; z_a = z0 - 0.5*g*t_a**2
    err  = np.abs(z_num[mask] - z_a)
    err  = np.where(err < 1e-16, 1e-16, err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    step = max(1, len(t_a)//20)
    ax1.plot(t_a, z_a, "k-", lw=2.0, label="Analytical")
    ax1.plot(t_a[::step], z_num[mask][::step], "rv", ms=6, label="Numerical",
             markerfacecolor="#d62728", markeredgecolor="k", markeredgewidth=0.4)
    ax1.set(xlabel="Time $t$ (s)", ylabel="Height $z$ (m)",
            title="(a) Free-fall trajectory", xlim=(0, t_hit))
    ax1.set_ylim(bottom=0); ax1.legend(loc="upper right")
    ax2.semilogy(t_a, err, "b-", lw=1.5)
    ax2.set(xlabel="Time $t$ (s)", title="(b) Numerical error",
            ylabel=r"$|z_\mathrm{num}-z_\mathrm{anal}|$ (m)", xlim=(0, t_hit))
    plt.tight_layout()
    plt.savefig("figures/fig1_freefall.pdf"); plt.savefig("figures/fig1_freefall.png")
    plt.close(); print("  fig1_freefall ✓")

def plot_error_vs_dt():
    z0, t_end = 8.0, 1.0
    dts = [1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4]
    errs = []
    for dt in dts:
        z, vz, t = z0, 0.0, 0.0
        for _ in range(int(t_end/dt)):
            vz += -g*dt; z += vz*dt; t += dt
        errs.append(abs(z - (z0-0.5*g*t**2)))
    dts = np.array(dts); errs = np.array(errs)
    ref = dts/dts[0]*errs[0]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.loglog(dts, errs, "bo-", label="Measured error")
    ax.loglog(dts, ref, "k--", label=r"$O(\Delta t)$ reference")
    ax.set(xlabel=r"Timestep $\Delta t$ (s)",
           ylabel=r"$|z_\mathrm{num}-z_\mathrm{anal}|$ at $t=1$ s (m)",
           title="Convergence: error vs timestep")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig2_error_vs_dt.pdf"); plt.savefig("figures/fig2_error_vs_dt.png")
    plt.close(); print("  fig2_error_vs_dt ✓")

def plot_bounce():
    df = pd.read_csv("test3_bounce.csv")
    t, z, KE = df["t"].values, df["z0"].values, df["KE"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax1.plot(t, z, "b-", lw=1.5, label="Particle centre")
    ax1.axhline(0.5, color="gray", ls="--", lw=1.2, label=r"Floor ($z=R$)")
    ax1.set(xlabel="Time $t$ (s)", ylabel="Height $z$ (m)", title="(a) Bounce trajectory")
    ax1.legend(); ax1.set_ylim(bottom=0)
    ax2.plot(t, KE, "r-", lw=1.5)
    ax2.set(xlabel="Time $t$ (s)", ylabel="Kinetic energy (J)", title="(b) KE with damping")
    plt.tight_layout()
    plt.savefig("figures/fig3_bounce.pdf"); plt.savefig("figures/fig3_bounce.png")
    plt.close(); print("  fig3_bounce ✓")

def plot_profiling():
    df = pd.read_csv("profile_N1000.csv")
    paired = sorted(zip(df["fraction"], df["phase"]), reverse=True)
    fracs, phases = zip(*paired)
    colors = ["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd"]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.barh(phases, fracs, color=colors[:len(phases)], edgecolor="k", lw=0.5)
    for bar, val in zip(bars, fracs):
        ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left", fontsize=FONTSIZE-1)
    ax.set(xlabel="Percentage of compute time (%)",
           title=r"Profiling: $N=1000$ particles, 1 thread", xlim=(0,110))
    ax.invert_yaxis(); ax.grid(axis="y", alpha=0)
    plt.tight_layout()
    plt.savefig("figures/fig4_profiling.pdf"); plt.savefig("figures/fig4_profiling.png")
    plt.close(); print("  fig4_profiling ✓")

def plot_ke_settling():
    if not os.path.exists("sim_N200.csv"):
        print("  fig5 skipped"); return
    df = pd.read_csv("sim_N200.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(df["t"], df["KE"], "b-", lw=1.5)
    ax.set(xlabel="Time $t$ (s)", ylabel="Total kinetic energy (J)",
           title=r"Kinetic energy vs time ($N=200$)")
    plt.tight_layout()
    plt.savefig("figures/fig5_ke_settling.pdf"); plt.savefig("figures/fig5_ke_settling.png")
    plt.close(); print("  fig5_ke_settling ✓")

def plot_speedup(csv_file, tag, note=""):
    if not os.path.exists(csv_file):
        print(f"  fig6 skipped ({csv_file} missing)"); return
    df = pd.read_csv(csv_file)
    colors  = {200:"#d62728", 1000:"#1f77b4", 5000:"#2ca02c"}
    markers = {200:"s",       1000:"D",        5000:"^"}
    labels  = {200:"$N=200$", 1000:"$N=1000$", 5000:"$N=5000$"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    thread_vals = sorted(df["threads"].unique())
    ideal = np.array(thread_vals, dtype=float)

    for N in sorted(df["N"].unique()):
        sub = df[df["N"]==N].sort_values("threads")
        T1  = float(sub[sub["threads"]==1]["Tp_ms"].values[0])
        Tp  = sub["Tp_ms"].values
        th  = sub["threads"].values
        sp  = T1 / Tp
        eff = sp / th
        ax1.plot(th, sp,  color=colors[N], marker=markers[N], label=labels[N], lw=1.5)
        ax2.plot(th, eff, color=colors[N], marker=markers[N], label=labels[N], lw=1.5)

    ax1.plot(thread_vals, ideal, "k--", lw=1.2, label="Ideal")
    ax2.axhline(1.0, color="k", ls="--", lw=1.2, label="Ideal")

    ax1.set(xlabel="Threads $p$", ylabel=r"Speedup $S(p)=T_1/T_p$",
            title="(a) Speedup"); ax1.set_xticks(thread_vals); ax1.legend(fontsize=9)
    ax2.set(xlabel="Threads $p$", ylabel=r"Efficiency $E(p)=S(p)/p$",
            title="(b) Parallel efficiency", ylim=(0, 1.15))
    ax2.set_xticks(thread_vals); ax2.legend(fontsize=9)

    if note:
        fig.text(0.5, -0.04, note, ha="center", fontsize=FONTSIZE-2, style="italic")

    plt.tight_layout()
    plt.savefig(f"figures/{tag}.pdf"); plt.savefig(f"figures/{tag}.png")
    plt.close(); print(f"  {tag} ✓")

def plot_snapshot():
    if not os.path.exists("sim_N200.csv"):
        return
    df = pd.read_csv("sim_N200.csv")
    row = df.iloc[-1]; cols = df.columns.tolist()
    xi = [c for c in cols if c.startswith("x") and c[1:].isdigit()]
    yi = [c for c in cols if c.startswith("y") and c[1:].isdigit()]
    zi = [c for c in cols if c.startswith("z") and c[1:].isdigit()]
    if not xi: return
    xv,yv,zv = row[xi].values.astype(float),row[yi].values.astype(float),row[zi].values.astype(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    sc1 = ax1.scatter(xv, yv, c=zv, cmap="viridis", s=40, edgecolors="k", lw=0.3)
    plt.colorbar(sc1, ax=ax1, label="$z$ (m)")
    ax1.set(xlabel="$x$ (m)", ylabel="$y$ (m)", title="(a) Top view (colour = $z$)")
    ax1.set_aspect("equal")
    sc2 = ax2.scatter(xv, zv, c=yv, cmap="plasma", s=40, edgecolors="k", lw=0.3)
    plt.colorbar(sc2, ax=ax2, label="$y$ (m)")
    ax2.set(xlabel="$x$ (m)", ylabel="$z$ (m)", title="(b) Side view (colour = $y$)")
    ax2.set_aspect("equal")
    plt.suptitle(r"Final particle configuration ($N=200$)", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig7_snapshot.pdf"); plt.savefig("figures/fig7_snapshot.png")
    plt.close(); print("  fig7_snapshot ✓")

if __name__ == "__main__":
    print("Generating figures...\n")
    plot_freefall()
    plot_error_vs_dt()
    plot_bounce()
    plot_profiling()
    plot_ke_settling()
    plot_speedup("scaling_results.csv", "fig6_speedup_efficiency",
                 note="Note: Run on 2-core sandbox. Replace with HPC data for final report.")
    plot_snapshot()
    print("\nAll figures saved to ./figures/")
