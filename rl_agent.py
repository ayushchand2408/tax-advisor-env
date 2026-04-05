"""
rl_agent.py — Q-Learning Agent for TaxAdvisorEnv
A real reinforcement learning agent that learns to file taxes
through trial and error, improving its score over episodes.
"""

import random
import json
import os
import numpy as np
from env import TaxAdvisorEnv, TaxAction, grade_task


# ─── Action Space ─────────────────────────────────────────────────────────────
# We define a fixed set of discrete actions the agent can take.
# This is what Q-Learning needs — a finite list of possible actions.

DEDUCTIBLE_CATEGORIES = [
    "home_office", "education", "health_expense",
    "charitable_donation", "mortgage_interest", "business_travel"
]
NON_DEDUCTIBLE_CATEGORIES = [
    "personal_food", "luxury_vacation", "personal_clothing", "gym_membership"
]
ALL_CATEGORIES = DEDUCTIBLE_CATEGORIES + NON_DEDUCTIBLE_CATEGORIES

# All possible actions the agent can choose from
ACTION_SPACE = (
    [{"tool_name": "get_income_data",  "arguments": {}}] +
    [{"tool_name": "get_receipts",     "arguments": {}}] +
    [{"tool_name": "search_tax_code",  "arguments": {"query": cat}} for cat in DEDUCTIBLE_CATEGORIES] +
    [{"tool_name": "classify_expense", "arguments": {"category": cat, "is_deductible": True}}  for cat in ALL_CATEGORIES] +
    [{"tool_name": "classify_expense", "arguments": {"category": cat, "is_deductible": False}} for cat in ALL_CATEGORIES] +
    [{"tool_name": "compute_taxes",    "arguments": {"income": 75000, "deductions": 0}}] +
    [{"tool_name": "compute_taxes",    "arguments": {"income": 75000, "deductions": 2600}}] +
    [{"tool_name": "compute_taxes",    "arguments": {"income": 120000, "deductions": 14200}}] +
    [{"tool_name": "fill_form_field",  "arguments": {"field": "taxpayer_name",   "value": "Alice Kumar"}}] +
    [{"tool_name": "fill_form_field",  "arguments": {"field": "filing_status",   "value": "single"}}] +
    [{"tool_name": "fill_form_field",  "arguments": {"field": "total_income",    "value": 75000}}] +
    [{"tool_name": "fill_form_field",  "arguments": {"field": "total_deductions","value": 2600}}] +
    [{"tool_name": "fill_form_field",  "arguments": {"field": "tax_owed",        "value": 12000}}] +
    [{"tool_name": "submit_form",      "arguments": {}}]
)
N_ACTIONS = len(ACTION_SPACE)


# ─── State Representation ─────────────────────────────────────────────────────

def state_to_key(state) -> str:
    """
    Convert environment state to a string key for the Q-table.
    The agent uses this to remember what it learned about each situation.
    """
    return (
        f"t{state.task_id}_"
        f"f{state.fields_filled}_"
        f"d{state.deductions_found}_"
        f"s{int(state.submitted)}_"
        f"steps{min(state.steps_taken, 15)}"  # cap steps to limit state space
    )


# ─── Q-Learning Agent ─────────────────────────────────────────────────────────

class QLearningAgent:
    """
    A Q-Learning agent that learns to file taxes through trial and error.

    Q-Learning works by maintaining a table of (state, action) → expected reward.
    The agent explores randomly at first, then exploits what it has learned.

    Key concepts:
    - Q-table: memory of how good each action is in each state
    - Epsilon: exploration rate (starts high, decays over time)
    - Alpha: learning rate (how fast to update Q-values)
    - Gamma: discount factor (how much to value future rewards)
    """

    def __init__(
        self,
        alpha: float = 0.3,    # learning rate
        gamma: float = 0.95,   # discount factor
        epsilon: float = 1.0,  # starting exploration rate
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[str, np.ndarray] = {}
        self.total_updates = 0

    def _get_q(self, state_key: str) -> np.ndarray:
        """Get Q-values for a state, initializing if unseen."""
        if state_key not in self.q_table:
            # Small random init breaks ties, helps exploration
            self.q_table[state_key] = np.random.uniform(-0.01, 0.01, N_ACTIONS)
        return self.q_table[state_key]

    def choose_action(self, state_key: str) -> int:
        """
        Epsilon-greedy action selection:
        - With probability epsilon: explore (random action)
        - Otherwise: exploit (best known action)
        """
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self._get_q(state_key)))

    def update(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
        done: bool,
    ):
        """
        Q-Learning update rule:
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))

        This is the core of Q-Learning — updating our estimate of how
        good an action was based on the reward we actually received.
        """
        q_vals = self._get_q(state_key)
        next_q_vals = self._get_q(next_state_key)

        # Bellman equation
        target = reward + (0 if done else self.gamma * np.max(next_q_vals))
        q_vals[action_idx] += self.alpha * (target - q_vals[action_idx])
        self.total_updates += 1

    def decay_epsilon(self):
        """Reduce exploration rate over time (agent becomes more confident)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save Q-table to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {k: v.tolist() for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump({
                "q_table": serializable,
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
            }, f)

    def load(self, path: str):
        """Load Q-table from disk."""
        with open(path) as f:
            data = json.load(f)
        self.q_table = {k: np.array(v) for k, v in data["q_table"].items()}
        self.epsilon = data["epsilon"]
        self.total_updates = data["total_updates"]


# ─── Training Loop ────────────────────────────────────────────────────────────

def run_episode(
    agent: QLearningAgent,
    task_id: int,
    max_steps: int = 25,
    train: bool = True,
) -> tuple[float, float]:
    """
    Run one episode and return (total_reward, grade_score).
    If train=True, updates the Q-table after each step.
    """
    env = TaxAdvisorEnv(task_id=task_id)
    env.reset()
    state_key = state_to_key(env.state())
    total_reward = 0.0

    for _ in range(max_steps):
        action_idx = agent.choose_action(state_key)
        action_data = ACTION_SPACE[action_idx]
        action = TaxAction(
            tool_name=action_data["tool_name"],
            arguments=action_data["arguments"],
        )

        _, reward, done, _ = env.step(action)
        total_reward += reward
        next_state_key = state_to_key(env.state())

        if train:
            agent.update(state_key, action_idx, reward, next_state_key, done)

        state_key = next_state_key
        if done:
            break

    if train:
        agent.decay_epsilon()

    return total_reward, grade_task(env)
