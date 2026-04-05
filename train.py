"""
train.py — Train the Q-Learning agent on TaxAdvisorEnv
Run: python train.py

This will:
1. Train the agent across all 3 tasks
2. Show real-time progress in the terminal
3. Save a learning curve plot to results/learning_curve.png
4. Save the trained agent to results/agent.json
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from rl_agent import QLearningAgent, run_episode

os.makedirs("results", exist_ok=True)

# ─── Training Config ──────────────────────────────────────────────────────────

EPISODES_PER_TASK = 600   # training episodes per task
EVAL_EVERY = 50           # evaluate (no exploration) every N episodes
EVAL_EPISODES = 10        # number of eval episodes to average

TASK_NAMES = {
    0: "Easy   (Tax Calculation)",
    1: "Medium (Deduction Finder)",
    2: "Hard   (Full Filing)",
}

# ─── Train ────────────────────────────────────────────────────────────────────

def train_task(task_id: int) -> dict:
    """Train one agent on one task and return training history."""
    print(f"\n{'='*55}")
    print(f"  Training Task {task_id}: {TASK_NAMES[task_id]}")
    print(f"{'='*55}")

    agent = QLearningAgent(
        alpha=0.3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.994,
    )

    history = {"episodes": [], "scores": [], "rewards": []}
    best_score = 0.0

    for ep in range(1, EPISODES_PER_TASK + 1):
        run_episode(agent, task_id=task_id, train=True)

        # Evaluate every EVAL_EVERY episodes
        if ep % EVAL_EVERY == 0:
            eval_scores = []
            eval_rewards = []
            for _ in range(EVAL_EPISODES):
                # eval: epsilon=0 (pure exploitation, no exploration)
                saved_eps = agent.epsilon
                agent.epsilon = 0.0
                reward, score = run_episode(agent, task_id=task_id, train=False)
                agent.epsilon = saved_eps
                eval_scores.append(score)
                eval_rewards.append(reward)

            avg_score = np.mean(eval_scores)
            avg_reward = np.mean(eval_rewards)
            history["episodes"].append(ep)
            history["scores"].append(avg_score)
            history["rewards"].append(avg_reward)

            if avg_score > best_score:
                best_score = avg_score

            bar = "█" * int(avg_score * 20) + "░" * (20 - int(avg_score * 20))
            print(
                f"  Ep {ep:4d}/{EPISODES_PER_TASK} | "
                f"Score: [{bar}] {avg_score:.3f} | "
                f"ε={agent.epsilon:.3f} | "
                f"States known: {len(agent.q_table)}"
            )

    # Save trained agent
    agent.save(f"results/agent_task{task_id}.json")
    print(f"\n  Best score: {best_score:.4f}")
    print(f"  Q-table size: {len(agent.q_table)} states learned")
    return history


# ─── Plot Learning Curves ─────────────────────────────────────────────────────

def plot_curves(histories: dict):
    """Plot learning curves for all 3 tasks on one chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "TaxAdvisorEnv — Q-Learning Agent Training Curves",
        fontsize=14, fontweight="bold", y=1.02
    )

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Task 0: Easy", "Task 1: Medium", "Task 2: Hard"]

    for i, (task_id, history) in enumerate(histories.items()):
        ax = axes[i]
        episodes = history["episodes"]
        scores = history["scores"]

        # Smooth the curve slightly
        smoothed = []
        window = 3
        for j in range(len(scores)):
            start = max(0, j - window)
            smoothed.append(np.mean(scores[start:j+1]))

        ax.plot(episodes, scores, color=colors[i], alpha=0.3, linewidth=1)
        ax.plot(episodes, smoothed, color=colors[i], linewidth=2.5, label=labels[i])
        ax.fill_between(episodes, 0, smoothed, color=colors[i], alpha=0.08)

        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
        ax.set_title(labels[i], fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(0, EPISODES_PER_TASK)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(True, alpha=0.2)

        # Annotate final score
        final = smoothed[-1] if smoothed else 0
        ax.annotate(
            f"Final: {final:.2f}",
            xy=(episodes[-1], final),
            xytext=(-60, 12),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=colors[i],
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5),
        )

    plt.tight_layout()
    path = "results/learning_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Learning curve saved to {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  TaxAdvisorEnv — Q-Learning RL Training")
    print("="*55)
    print(f"  Episodes per task : {EPISODES_PER_TASK}")
    print(f"  Eval every        : {EVAL_EVERY} episodes")
    print(f"  Tasks             : 3 (easy / medium / hard)")

    histories = {}
    final_scores = {}

    for task_id in [0, 1, 2]:
        history = train_task(task_id)
        histories[task_id] = history
        final_scores[task_id] = history["scores"][-1] if history["scores"] else 0.0

    # Plot
    plot_curves(histories)

    # Final summary
    print("\n" + "="*55)
    print("  TRAINING COMPLETE — FINAL SCORES")
    print("="*55)
    task_labels = ["Easy  ", "Medium", "Hard  "]
    for task_id, label in enumerate(task_labels):
        score = final_scores[task_id]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  Task {task_id} {label}  [{bar}]  {score:.4f}")

    overall = np.mean(list(final_scores.values()))
    print(f"\n  Overall avg: {overall:.4f}")
    print("\n  Trained agents saved to results/")
    print("  Learning curve saved to results/learning_curve.png")
    print("\n  Add results/learning_curve.png to your README!")


if __name__ == "__main__":
    main()
