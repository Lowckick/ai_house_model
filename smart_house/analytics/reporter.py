"""
Analytics and reporting module.
Reads training DB and generates plots + summary reports.
"""

import os
import sqlite3
import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "training.db")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reports")


def load_episode_summaries(db_path: str) -> list:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT episode, total_reward, avg_comfort, avg_power FROM episode_summary ORDER BY episode"
    ).fetchall()
    conn.close()
    return rows


def load_step_data(db_path: str, episode: Optional[int] = None) -> list:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    if episode is not None:
        rows = conn.execute(
            "SELECT step, reward, comfort, temperature, humidity, co2, pm25, dirt, power_kw, hvac_mode, cleaner_bot "
            "FROM episodes WHERE episode=? ORDER BY step",
            (episode,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT step, reward, comfort, temperature, humidity, co2, pm25, dirt, power_kw, hvac_mode, cleaner_bot "
            "FROM episodes ORDER BY episode, step"
        ).fetchall()
    conn.close()
    return rows


def generate_training_report(db_path: str = DB_PATH, output_dir: str = REPORT_DIR):
    """Generate matplotlib plots and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    summaries = load_episode_summaries(db_path)

    if not summaries:
        print("[Reporter] No data to report yet.")
        return

    episodes = [r[0] for r in summaries]
    rewards = [r[1] for r in summaries]
    comforts = [r[2] for r in summaries]
    powers = [r[3] for r in summaries]

    # Smooth with rolling average
    def smooth(data, w=10):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode="valid").tolist()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Smart House ML Agent — Training Report", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Total reward per episode
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, rewards, alpha=0.3, color="steelblue", label="raw")
    if len(rewards) >= 10:
        s = smooth(rewards)
        ax1.plot(range(1, len(s) + 1), s, color="steelblue", linewidth=2, label="MA-10")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()

    # 2. Average comfort
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, comforts, alpha=0.3, color="green", label="raw")
    if len(comforts) >= 10:
        s = smooth(comforts)
        ax2.plot(range(1, len(s) + 1), s, color="green", linewidth=2, label="MA-10")
    ax2.set_ylim(0, 100)
    ax2.set_title("Average Comfort Score (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Comfort %")
    ax2.legend()

    # 3. Average power usage
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(episodes, powers, alpha=0.3, color="orange", label="raw")
    if len(powers) >= 10:
        s = smooth(powers)
        ax3.plot(range(1, len(s) + 1), s, color="orange", linewidth=2, label="MA-10")
    ax3.set_title("Average Power Usage (kW)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("kW")
    ax3.legend()

    # 4. Last episode detailed: comfort + temp
    if summaries:
        last_ep = summaries[-1][0]
        steps_data = load_step_data(db_path, last_ep)
        if steps_data:
            steps = [r[0] for r in steps_data]
            ep_comfort = [r[2] for r in steps_data]
            ep_temp = [r[3] for r in steps_data]
            ep_co2 = [r[5] for r in steps_data]
            ep_dirt = [r[7] for r in steps_data]

            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(steps, ep_comfort, color="green")
            ax4.set_ylim(0, 100)
            ax4.set_title(f"Ep {last_ep}: Comfort over Time")
            ax4.set_xlabel("Step (×10 min)")
            ax4.set_ylabel("Comfort %")

            ax5 = fig.add_subplot(gs[1, 1])
            ax5.plot(steps, ep_temp, color="red", label="Indoor Temp")
            ax5.axhline(22, color="green", linestyle="--", linewidth=0.8, label="Target 22°C")
            ax5.set_title(f"Ep {last_ep}: Temperature (°C)")
            ax5.set_xlabel("Step (×10 min)")
            ax5.set_ylabel("°C")
            ax5.legend()

            ax6 = fig.add_subplot(gs[1, 2])
            ax6l = ax6
            ax6r = ax6.twinx()
            ax6l.plot(steps, ep_co2, color="purple", label="CO₂ ppm")
            ax6r.plot(steps, ep_dirt, color="brown", linestyle="--", label="Dirt %")
            ax6l.set_title(f"Ep {last_ep}: CO₂ & Dirt")
            ax6l.set_xlabel("Step")
            ax6l.set_ylabel("CO₂ (ppm)", color="purple")
            ax6r.set_ylabel("Dirt (%)", color="brown")

    out_path = os.path.join(output_dir, f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Reporter] Report saved to: {out_path}")
    return out_path


def print_text_summary(db_path: str = DB_PATH):
    """Print a quick text summary of training progress."""
    summaries = load_episode_summaries(db_path)
    if not summaries:
        print("[Reporter] No training data found.")
        return

    print("\n" + "=" * 60)
    print("  SMART HOUSE ML — TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total episodes: {len(summaries)}")

    last10 = summaries[-10:]
    avg_reward = sum(r[1] for r in last10) / len(last10)
    avg_comfort = sum(r[2] for r in last10) / len(last10)
    avg_power = sum(r[3] for r in last10) / len(last10)

    print(f"  Last 10 episodes:")
    print(f"    Avg reward:  {avg_reward:+.2f}")
    print(f"    Avg comfort: {avg_comfort:.1f}%")
    print(f"    Avg power:   {avg_power:.3f} kW")
    print("=" * 60 + "\n")
