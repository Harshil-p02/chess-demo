import streamlit as st

import chess
import chess.pgn
import chess.svg

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import asyncio
import copy
import chess
import chess.engine
import chess.pgn
import time
import sys

import os
import os.path
import pickle

from torchtext.legacy.data import Field

import transformer

SOS_WORD = "<sos>"
EOS_WORD = "<eos>"
BLANK_WORD = "<blank>"
MAX_LEN = 15 * 8
path = "chpts/0/transformer-best.pt"


# print(f'The model has {count_param(model):,} trainable parameters')

# def load_model(model):
#
#     print(f"Loading previous model state from {path}...")
#     model.load_state_dict(torch.load(path, map_location="cuda:0"))
#     print("Model loaded successfully!")
#
# load_model(model)

def predict(model, input_sequence, max_length=MAX_LEN, SOS_token=2, EOS_token=3):
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=st.session_state.device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(st.session_state.device)

        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=st.session_state.device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


def render_svg(svg):
    html = f'<html>{svg}</html>'
    st.write(html, unsafe_allow_html=True)

if "game" not in st.session_state:
    st.session_state.game = chess.Board()

if "moves" not in st.session_state:
    st.session_state.moves = []

if "cnt" not in st.session_state:
    st.session_state.cnt = 0

if "device" not in st.session_state:
    st.session_state.device = "cuda:0" if torch.cuda.is_available() else "cpu"

if "SRC" not in st.session_state:
    st.session_state.SRC = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
    st.session_state.SRC.vocab = torch.load("moves-vocab.pt")

if "TGT" not in st.session_state:
    st.session_state.TGT = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
    st.session_state.TGT.vocab = torch.load("moves-vocab.pt")

if "model" not in st.session_state:
    st.session_state.model = transformer.Transformer(num_tokens_inp=len(st.session_state.SRC.vocab), num_tokens_out=len(st.session_state.TGT.vocab), dim_model=512, num_heads=8,
                                    num_encoder_layers=3, num_decoder_layers=3, max_len=MAX_LEN, dropout_p=0.1).to(st.session_state.device)
    st.session_state.model.load_state_dict(torch.load(path, map_location=st.session_state.device))

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

def model_move(prediction):
    move = "".join(prediction[2:-2])
    st.session_state.game.push_san(move)
    render_svg(chess.svg.board(st.session_state.game))

def run(move):
    if move is not None:
        tokenised_move = tokeniser(move)
        st.session_state.game.push_san(move)

        src = []
        for i in tokenised_move:
            src.append(st.session_state.SRC.vocab.stoi[i])
        src = torch.tensor(list(map(int, src))).reshape(1, -1).to(st.session_state.device)

        # time.sleep(1.5)

        pred = predict(st.session_state.model, src)

        prediction = []
        for i in pred:
            prediction.append(st.session_state.TGT.vocab.itos[i])
            print(st.session_state.TGT.vocab.itos[i], end=" ")

        model_move(prediction)
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


