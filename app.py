# app.py
import streamlit as st
from game import TicTacToe
from q_learning import MinimaxEnvironment

st.set_page_config(page_title="Tic-Tac-Toe RL", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Board cells: secondary buttons become equal squares with no border */
[data-testid="baseButton-secondary"] {
    aspect-ratio: 1 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 0 !important;
    background: #1c1c1c !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 2rem !important;
    color: #d0d0d0 !important;
    line-height: 1 !important;
    box-shadow: none !important;
}
[data-testid="baseButton-secondary"]:hover:not(:disabled) {
    background: #262626 !important;
    border: none !important;
}
[data-testid="baseButton-secondary"]:disabled {
    opacity: 1 !important;
    cursor: default !important;
    background: #1c1c1c !important;
    border: none !important;
}

/* Action buttons (train, play, evaluate) */
[data-testid="baseButton-primary"] {
    background: #1c1c1c !important;
    border: 1px solid #333 !important;
    color: #c0c0c0 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}
[data-testid="baseButton-primary"]:hover:not(:disabled) {
    background: #242424 !important;
    border-color: #4a4a4a !important;
    color: #d8d8d8 !important;
}

/* Tighten column gaps for the board rows */
[data-testid="stHorizontalBlock"] {
    gap: 6px !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: #1c1c1c !important;
    border-color: #333 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_winning_cells(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return [(i, 0), (i, 1), (i, 2)]
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != 0:
            return [(0, j), (1, j), (2, j)]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return [(0, 0), (1, 1), (2, 2)]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return [(0, 2), (1, 1), (2, 0)]
    return None


def render_board_html(board, winning_cells=None):
    """Render the board as a fixed-size HTML/CSS grid with an optional SVG winning line."""
    CELL, GAP, PAD = 96, 8, 8
    # Total = 3*96 + 2*8 + 2*8 = 320
    TOTAL = 3 * CELL + 2 * GAP + 2 * PAD
    SYMS = {0: "", 1: "❌", 2: "⭕"}

    # Cell-center coordinates inside the SVG (which starts at the outer div's top-left)
    def cx(c): return PAD + c * (CELL + GAP) + CELL // 2
    def cy(r): return PAD + r * (CELL + GAP) + CELL // 2

    cells = ""
    for r in range(3):
        for c in range(3):
            is_win = winning_cells and (r, c) in winning_cells
            bg = "#262626" if is_win else "#1c1c1c"
            cells += (
                f'<div style="width:{CELL}px;height:{CELL}px;background:{bg};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:2rem;border-radius:3px;color:#d0d0d0;">'
                f'{SYMS[board[r][c]]}</div>'
            )

    line = ""
    if winning_cells:
        r0, c0 = winning_cells[0]
        r2, c2 = winning_cells[2]
        line = (
            f'<svg style="position:absolute;top:0;left:0;'
            f'width:{TOTAL}px;height:{TOTAL}px;pointer-events:none;z-index:10;">'
            f'<line x1="{cx(c0)}" y1="{cy(r0)}" x2="{cx(c2)}" y2="{cy(r2)}"'
            f' stroke="#d0d0d0" stroke-width="5" stroke-linecap="round" opacity="0.85"/>'
            f'</svg>'
        )

    return (
        f'<div style="position:relative;display:inline-block;'
        f'width:{TOTAL}px;height:{TOTAL}px;">'
        f'<div style="display:grid;grid-template-columns:repeat(3,{CELL}px);'
        f'grid-template-rows:repeat(3,{CELL}px);'
        f'gap:{GAP}px;padding:{PAD}px;background:#2a2a2a;border-radius:5px;">'
        f'{cells}</div>'
        f'{line}'
        f'</div>'
    )


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ('env', None), ('game', None),
    ('training_done', False), ('game_started', False),
    ('eval_accuracy', None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Tic-Tac-Toe RL")
left_col, right_col = st.columns([0.6, 0.4])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Board
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.subheader("Board")

    if not st.session_state.training_done:
        st.info("Train the AI first using the panel on the right.")

    elif not st.session_state.game_started:
        st.info("Click PLAY to start.")

    else:
        game = st.session_state.game
        if game:
            result = game.get_game_result()

            if result is not None:
                # ── Game over: static HTML board with winning line ────────────
                wc = get_winning_cells(game.board) if result in (1, 2) else None
                st.markdown(render_board_html(game.board, wc), unsafe_allow_html=True)
                st.write("")
                if result == 1:
                    st.success("You win.")
                elif result == 2:
                    st.error("AI wins.")
                else:
                    st.warning("Draw.")
                if st.button("Play Again", type="primary"):
                    st.session_state.game = TicTacToe()
                    st.rerun()

            else:
                # ── Active game: interactive 3×3 button grid ─────────────────
                st.caption("Your turn  ❌")

                # Constrain board to middle third of the left column
                _, board_area, _ = st.columns([1, 1, 1])
                with board_area:
                    for i in range(3):
                        row_cols = st.columns(3, gap="small")
                        for j in range(3):
                            pos = i * 3 + j
                            with row_cols[j]:
                                if game.board[i][j] == 0:
                                    if st.button(
                                        " ",           # non-breaking space — empty cell
                                        key=f"p{pos}",
                                        use_container_width=True,
                                    ):
                                        game.make_move(pos, 1)
                                        if game.get_game_result() is None:
                                            s = st.session_state.env.agent_o.board_to_state(game.board)
                                            a = st.session_state.env.agent_o.get_action(s, training=False)
                                            if a is not None:
                                                game.make_move(a[0] * 3 + a[1], 2)
                                        st.rerun()
                                else:
                                    sym = "❌" if game.board[i][j] == 1 else "⭕"
                                    st.button(
                                        sym,
                                        key=f"p{pos}d",
                                        disabled=True,
                                        use_container_width=True,
                                    )

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Control Panel
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    st.subheader("Control Panel")
    st.write("**1. Train AI**")

    num_games = st.number_input(
        "Games to train:", min_value=10, max_value=50000, value=10000, step=10
    )

    if st.button("Train AI", type="primary", use_container_width=True):
        st.session_state.env = MinimaxEnvironment()
        st.session_state.eval_accuracy = None
        pbar = st.progress(0)
        status = st.empty()

        def update_progress(current, total):
            pbar.progress(current / total)
            status.text(f"Training: {current} / {total}")

        st.session_state.env.train(int(num_games), progress_callback=update_progress)
        st.session_state.training_done = True
        pbar.empty()
        status.empty()
        st.success(f"Done — {int(num_games)} games trained.")
        st.rerun()

    if st.session_state.training_done and st.session_state.env:
        st.divider()

        if st.button("PLAY", type="primary", use_container_width=True):
            st.session_state.game = TicTacToe()
            st.session_state.game_started = True
            st.rerun()

        st.divider()
        st.write("**2. Evaluate AI**")

        if st.button("Policy accuracy  (100 games)", type="primary", use_container_width=True):
            acc = st.session_state.env.evaluate_policy_accuracy(100)
            st.session_state.eval_accuracy = acc
            st.rerun()

        if st.session_state.eval_accuracy is not None:
            acc = st.session_state.eval_accuracy
            pct = acc['accuracy']
            color = "success" if pct >= 90 else ("warning" if pct >= 70 else "error")
            getattr(st, color)(f"{pct:.1f}% optimal moves")
            st.caption(f"{acc['optimal_moves']} / {acc['total_moves']} moves matched minimax")
