import collections
import numpy as np
import random
from game import TicTacToe


class QLearner:
    def __init__(self, player, alpha=0.8, gamma=0.9, eps=0.1):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = {(i, j): collections.defaultdict(float) for i in range(3) for j in range(3)}

    def board_to_state(self, board):
        return "".join(
            "-" if board[i, j] == 0 else ("X" if board[i, j] == 1 else "O")
            for i in range(3) for j in range(3)
        )

    def get_possible_actions(self, state):
        return [(i, j) for i in range(3) for j in range(3) if state[i * 3 + j] == "-"]

    def get_action(self, state, training=True):
        actions = self.get_possible_actions(state)
        if not actions:
            return None
        if training and random.random() < self.eps:
            return random.choice(actions)
        values = np.array([self.Q[a][state] for a in actions])
        return actions[np.random.choice(np.where(values == values.max())[0])]

    def update(self, s, s_, a, r):
        if s_ is not None:
            next_actions = self.get_possible_actions(s_)
            max_q_next = max((self.Q[a][s_] for a in next_actions), default=0.0)
            self.Q[a][s] += self.alpha * (r + self.gamma * max_q_next - self.Q[a][s])
        else:
            self.Q[a][s] += self.alpha * (r - self.Q[a][s])

    def set_eps(self, eps):
        self.eps = eps


class MinimaxEnvironment:
    def __init__(self):
        self.agent_o = QLearner(player=2, alpha=0.5, gamma=0.95, eps=1.0)
        self.exact_memo = {}
        self._minimax_exact(np.zeros((3, 3), dtype=int), True)

    # ── Board helpers ─────────────────────────────────────────────────────────

    def get_available_moves(self, board):
        return [i * 3 + j for i in range(3) for j in range(3) if board[i][j] == 0]

    def make_move_copy(self, board, position, player):
        b = board.copy()
        b[position // 3][position % 3] = player
        return b

    def check_game_result(self, board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 0:
                return board[i][0]
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] != 0:
                return board[0][j]
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]
        return 0 if np.all(board != 0) else None

    # ── Exact minimax (no alpha-beta — avoids cached-bound corruption) ────────

    def _minimax_exact(self, board, is_x_turn):
        key = (tuple(board.flatten()), is_x_turn)
        if key in self.exact_memo:
            return self.exact_memo[key]
        result = self.check_game_result(board)
        if result is not None:
            v = 1 if result == 1 else (-1 if result == 2 else 0)
            self.exact_memo[key] = v
            return v
        moves = self.get_available_moves(board)
        if is_x_turn:
            v = max(self._minimax_exact(self.make_move_copy(board, m, 1), False) for m in moves)
        else:
            v = min(self._minimax_exact(self.make_move_copy(board, m, 2), True) for m in moves)
        self.exact_memo[key] = v
        return v

    def get_minimax_move(self, board, player=1):
        best_move, best_val = None, float('-inf') if player == 1 else float('inf')
        p, next_turn = (1, False) if player == 1 else (2, True)
        for move in self.get_available_moves(board):
            v = self._minimax_exact(self.make_move_copy(board, move, p), next_turn)
            if (player == 1 and v > best_val) or (player == 2 and v < best_val):
                best_val, best_move = v, move
        return best_move

    # ── Training ──────────────────────────────────────────────────────────────

    def play_game(self, training=True, x_random_rate=0.0):
        game = TicTacToe()
        move_sequence = []
        current_player = 1

        while game.get_game_result() is None:
            if current_player == 1:
                if x_random_rate > 0.0 and random.random() < x_random_rate:
                    action = random.choice(self.get_available_moves(game.board))
                else:
                    action = self.get_minimax_move(game.board, player=1)
                move_sequence.append((self.agent_o.board_to_state(game.board), action, None))
                game.make_move(action, 1)
                current_player = 2
            else:
                state = self.agent_o.board_to_state(game.board)
                action_tuple = self.agent_o.get_action(state, training=training)
                if action_tuple is None:
                    break
                move_sequence.append((state, action_tuple, self.agent_o))
                game.make_move(action_tuple[0] * 3 + action_tuple[1], 2)
                current_player = 1

        result = game.get_game_result()
        if training:
            self._update_agent(move_sequence, result)
        return result

    def _update_agent(self, move_sequence, result):
        # next_state for each O move must be the board state AFTER X has responded
        # (start of O's next turn), not immediately after O placed its piece.
        # Intermediate O moves get r=0; only the final move gets the terminal reward.
        reward_o = 1.0 if result == 2 else (0.5 if result == 0 else -1.0)
        o_moves = [(idx, s, a, ag) for idx, (s, a, ag) in enumerate(move_sequence) if ag is not None]
        for i, (seq_idx, state, action, agent) in enumerate(o_moves):
            if i < len(o_moves) - 1:
                next_state = move_sequence[o_moves[i + 1][0]][0]
                r = 0.0
            else:
                next_state = None
                r = reward_o
            agent.update(state, next_state, action, r)

    def train(self, num_games, progress_callback=None):
        decay_games = max(1, int(0.05 * num_games))
        for game_num in range(num_games):
            self.agent_o.set_eps(max(0.01, 1.0 - (game_num / decay_games)))
            self.play_game(training=True, x_random_rate=1.0 - (game_num / num_games))
            if progress_callback is not None:
                progress_callback(game_num + 1, num_games)
        self.agent_o.set_eps(0.0)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_policy_accuracy(self, num_games=100):
        self.agent_o.set_eps(0.0)
        total_moves = optimal_moves = 0

        for _ in range(num_games):
            game = TicTacToe()
            current_player = 1
            while game.get_game_result() is None:
                if current_player == 1:
                    game.make_move(random.choice(self.get_available_moves(game.board)), 1)
                    current_player = 2
                else:
                    state = self.agent_o.board_to_state(game.board)
                    action_tuple = self.agent_o.get_action(state, training=False)
                    if action_tuple is None:
                        break
                    agent_flat = action_tuple[0] * 3 + action_tuple[1]
                    best_flat = self.get_minimax_move(game.board, player=2)
                    best_val = self._minimax_exact(self.make_move_copy(game.board, best_flat, 2), True)
                    agent_val = self._minimax_exact(self.make_move_copy(game.board, agent_flat, 2), True)
                    total_moves += 1
                    if agent_val == best_val:
                        optimal_moves += 1
                    game.make_move(agent_flat, 2)
                    current_player = 1

        accuracy = (optimal_moves / total_moves * 100) if total_moves > 0 else 0.0
        return {'total_moves': total_moves, 'optimal_moves': optimal_moves, 'accuracy': accuracy}
