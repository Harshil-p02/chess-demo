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
    
def tokeniser(pgn_move):
    chars = list(pgn_move)

    move = "[ "
    if len(chars) == 2:
        move += ''.join(chars) + " "
    elif chars[0] == "O":
        if len(chars) == 3 or len(chars) == 5:
            move += ''.join(chars) + " "
        else:
            if len(chars) == 4:
                move += "O-O " + chars[3] + " "
            else:
                move += "O-O-O " + chars[5] + " "
    else:
        j = 0
        while j < len(chars):
            if j + 1 < len(chars) and chars[j + 1].isnumeric():
                move += ''.join(chars[j:j + 2]) + " "
                j += 1
            else:
                move += chars[j] + " "
            j += 1
    move = move.strip()
    move += " ]"
    return move

def run(move):
    tokenised_move = tokeniser(move)
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


