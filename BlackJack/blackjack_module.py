import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
import os


def build_index_map(values):
    """Return a dictionary mapping element -> index for a list of values."""
    return {val: idx for idx, val in enumerate(values)}


class MonteCarloBlackjack:
    """
    Blackjack simulator with Monte Carlo control (epsilon-greedy, constant-alpha).

    Workflow:
        bj = MonteCarloBlackjack(num_episodes=1_000_000, epsilon=0.1, alpha=1/5000, seed=42)
        bj.train()
        bj.plot("Q")

    - States: (player_sum, dealer_card, has_usable_ace)
    - Actions: Stick ("S") or Hit ("H")
    - Rewards: +1 win, 0 draw, -1 loss
    """

    # Define the environment setup
    card_deck = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    player_values = list(range(12, 22))  # 12–21
    dealer_upcards = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ace_states = [False, True]
    legal_actions = ["S", "H"]

    # Create mappings for indexing
    player_map, dealer_map, ace_map, action_map, card_map = map(
        build_index_map,
        [player_values, dealer_upcards, ace_states, legal_actions, card_deck]
    )

    def __init__(self, num_episodes, epsilon, alpha, seed=0):
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.seed = seed
        self.Q = None
        self.C = None
        self.history = None
        self._reset_tables()

    def _reset_tables(self):
        """Initialize Q-values and visitation counters."""
        self.Q = np.zeros(
            (
                len(self.player_values),
                len(self.dealer_upcards),
                len(self.ace_states),
                len(self.legal_actions)
            )
        )
        self.C = np.zeros_like(self.Q, dtype=int)
        os.makedirs("results", exist_ok=True)

    @staticmethod
    def draw_cards(n):
        """Sample n cards from an infinite deck."""
        cards = np.random.choice(MonteCarloBlackjack.card_deck, size=n, replace=True)
        return [c if c == "A" else int(c) for c in cards]

    @staticmethod
    def compute_value(hand):
        """Return (sum, usable_ace_flag) for a given hand."""
        base_sum = sum(c for c in hand if c != "A")
        ace_count = hand.count("A")

        if ace_count == 0:
            return base_sum, False

        soft_sum = base_sum + 10 + ace_count  # one ace = 11
        hard_sum = base_sum + ace_count       # all aces = 1
        if soft_sum > 21:
            return hard_sum, False
        return soft_sum, True

    def _state_to_indices(self, player_total, dealer_card, ace_flag):
        return (
            self.player_map[player_total],
            self.dealer_map[dealer_card],
            self.ace_map[ace_flag],
        )

    def epsilon_greedy(self, player_total, dealer_card, ace_flag):
        """Choose action using epsilon-greedy rule."""
        i, j, k = self._state_to_indices(player_total, dealer_card, ace_flag)
        greedy_action = np.argmax(self.Q[i, j, k, :])

        if np.random.rand() < self.epsilon:
            return self.legal_actions[1 - greedy_action]  # flip action
        return self.legal_actions[greedy_action]

    def play_episode(self):
        """Simulate a full blackjack episode and return trajectory + outcome."""
        # Initial cards
        player = self.draw_cards(2)
        dealer_up = self.draw_cards(1)[0]

        states, actions = [[player, dealer_up]], []

        # Force player sum ≥ 12
        while True:
            total, _ = self.compute_value(states[-1][0])
            if total < 12:
                states.append([states[-1][0] + self.draw_cards(1), dealer_up])
                actions.append("H")
            else:
                break

        # Player actions
        while True:
            total, ace_flag = self.compute_value(states[-1][0])
            action = self.epsilon_greedy(total, dealer_up, ace_flag)
            actions.append(action)

            if action == "S":
                states.append(states[-1])
                break
            else:
                next_hand = states[-1][0] + self.draw_cards(1)
                states.append([next_hand, dealer_up])
                new_total, _ = self.compute_value(next_hand)
                if new_total > 21:
                    return states, actions, -1, "player_bust"

        # Dealer actions
        dealer_hand = [dealer_up] + self.draw_cards(1)
        while True:
            dealer_total, _ = self.compute_value(dealer_hand)
            if dealer_total > 21:
                return states, actions, 1, "dealer_bust"
            if dealer_total >= 17:
                break
            dealer_hand += self.draw_cards(1)

        # Compare outcomes
        player_total, _ = self.compute_value(states[-1][0])
        dealer_total, _ = self.compute_value(dealer_hand)

        if player_total > dealer_total:
            return states, actions, 1, "win"
        elif player_total < dealer_total:
            return states, actions, -1, "loss"
        return states, actions, 0, "draw"

    def train(self, record_history=True):
        """Run Monte Carlo control algorithm."""
        self._reset_tables()
        np.random.seed(self.seed)

        snapshots = []
        checkpoints = list(range(0, self.num_episodes + 1, 1000))

        for ep in tqdm(range(self.num_episodes + 1)):
            states, actions, reward, _ = self.play_episode()

            if record_history and ep in checkpoints:
                snapshots.append(self._to_dataframe("Q").assign(episode=ep))

            for (hand, dealer_up), action in zip(states[:-1], actions):
                total, ace_flag = self.compute_value(hand)
                if total < 12:
                    continue

                i, j, k = self._state_to_indices(total, dealer_up, ace_flag)
                a = self.action_map[action]

                q_old = self.Q[i, j, k, a]
                self.Q[i, j, k, a] = q_old + self.alpha * (reward - q_old)
                self.C[i, j, k, a] += 1

        if record_history:
            self.history = pd.concat(snapshots, axis=0)

        self._save_results()

    def _file_names(self):
        tag = f"E{self.num_episodes}__eps{str(self.epsilon).replace('.','_')}__a{str(self.alpha).replace('.','_')}__s{self.seed}"
        return f"results/Q_{tag}", f"results/C_{tag}", f"results/Qhist_{tag}"

    def _save_results(self):
        Qfile, Cfile, Hfile = self._file_names()
        np.save(Qfile, self.Q)
        np.save(Cfile, self.C)
        if self.history is not None:
            self.history.to_pickle(Hfile + ".pkl")

    def load_results(self):
        Qfile, Cfile, Hfile = self._file_names()
        self.Q = np.load(Qfile + ".npy")
        self.C = np.load(Cfile + ".npy")
        try:
            self.history = pd.read_pickle(Hfile + ".pkl")
        except FileNotFoundError:
            pass

    def _to_dataframe(self, which="Q"):
        """Convert Q or C array into a flat dataframe for plotting."""
        arr = getattr(self, which)
        frames = []

        for ace_idx, ace_flag in enumerate(self.ace_states):
            for act_idx, act in enumerate(self.legal_actions):
                slice_arr = arr[:, :, ace_idx, act_idx]
                df = pd.DataFrame(
                    slice_arr,
                    index=self.player_values,
                    columns=self.dealer_upcards
                ).stack().to_frame(which)

                if which == "Q":
                    optimal = arr.argmax(-1)[:, :, ace_idx] == act_idx
                    opt_df = pd.DataFrame(
                        optimal.astype(int),
                        index=self.player_values,
                        columns=self.dealer_upcards
                    ).stack().to_frame("optimal")
                    df = pd.concat([df, opt_df], axis=1)

                df = df.reset_index().rename(
                    columns={"level_0": "player_sum", "level_1": "dealer_card"}
                )
                df["usable_ace"] = ace_flag
                df["action"] = act
                frames.append(df)

        return pd.concat(frames, axis=0)

    def plot(self, which="Q", width=300, height=200):
        """Generate heatmap of Q or C values."""
        df = self._to_dataframe(which)
        val_col = "Q" if "Q" in df.columns else "C"
        color = alt.Color(val_col, scale=alt.Scale(domain=[-1, 1])) if val_col == "Q" else alt.Color(val_col)

        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(x="player_sum:O", y="dealer_card:O", color=color)
        )

        df["val_round"] = df[val_col].round(2)
        labels = alt.Chart(df).mark_text().encode(
            x="player_sum:O", y="dealer_card:O", text="val_round"
        )

        if which == "Q":
            opt_boxes = (
                alt.Chart(df)
                .mark_rect(fill=None, stroke="green", strokeWidth=2)
                .encode(x="player_sum:O", y="dealer_card:O")
                .transform_filter(alt.datum.optimal == 1)
            )
            chart = heatmap + labels + opt_boxes
        else:
            chart = heatmap + labels

        return chart.properties(width=width, height=height).facet(
            row="action", column="usable_ace"
        )
