import streamlit as st

import chess
import chess.pgn
import chess.svg


def render_svg(svg):
    html = f'<html>{svg}</html>'
    st.write(html, unsafe_allow_html=True)

if "game" not in st.session_state:
    st.session_state.game = chess.Board()

if "moves" not in st.session_state:
    st.session_state.moves = []

if "cnt" not in st.session_state:
    st.session_state.cnt = 0

def run(move):
    if move is not None:
        st.session_state.game.push_san(move)

    render_svg(chess.svg.board(st.session_state.game))

def get_move():
    if st.session_state.cnt > 1:
        st.session_state.moves.append(move)
        st.write(" ".join(st.session_state.moves))
        run(st.session_state.moves[-1])
    else:
        run(None)


new_move = st.button(label="submit")

move = st.text_input(label="Enter PGN move:", placeholder="move")
if new_move:
    st.session_state.cnt += 1
    get_move()
#     # st.write(move)
#     # prev_move = move
#     st.session_state.moves.append(move)
#     print(st.session_state.moves)
#     st.write(st.session_state.moves)
    # run(move)


