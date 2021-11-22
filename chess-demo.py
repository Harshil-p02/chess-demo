import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import asyncio
import copy
import chess
import chess.engine
import chess.pgn

import sys

import os
import os.path
import pickle

from torchtext.legacy.data import Field

import transformer


SOS_WORD = "<sos>"
EOS_WORD = "<eos>"
BLANK_WORD = "<blank>"
MAX_LEN = 15*8

SRC = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
TGT = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
# fields = (("src", SRC), ("tgt", TGT))


SRC.vocab = torch.load("moves-vocab.pt")
TGT.vocab = torch.load("moves-vocab.pt")

print(len(SRC.vocab))
print(SRC.vocab.itos[:len(SRC.vocab)])
print()

print(len(TGT.vocab))
print(TGT.vocab.itos[:len(TGT.vocab)])
print()


# BATCH_SIZE = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

INP = len(SRC.vocab)
OUT = len(TGT.vocab)
model = transformer.Transformer(num_tokens_inp=INP, num_tokens_out=OUT, dim_model=512, num_heads=8,
								num_encoder_layers=3, num_decoder_layers=3, max_len=MAX_LEN, dropout_p=0.1).to(device)

# opt = optim.Adam(model.parameters(), lr=0.0001)

TGT_PAD_IDX = TGT.vocab.stoi[BLANK_WORD]
print(f"ignoring pad word: {BLANK_WORD} with index {TGT_PAD_IDX} in vocab")
loss_fn = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)


def count_param(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_param(model):,} trainable parameters')


def load_model(model):
	path = "chpts/0/transformer-best.pt"
	print(f"Loading previous model state from {path}...")
	model.load_state_dict(torch.load(path, map_location="cuda:0"))
	print("Model loaded successfully!")

load_model(model)
print()

def predict(model, input_sequence, max_length=MAX_LEN, SOS_token=2, EOS_token=3):
	model.eval()

	y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

	for _ in range(max_length):
		# Get source mask
		tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

		pred = model(input_sequence, y_input, tgt_mask)

		next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
		next_item = torch.tensor([[next_item]], device=device)

		# Concatenate previous input with predicted best word
		y_input = torch.cat((y_input, next_item), dim=1)

		# Stop if model predicts end of sentence
		if next_item.view(-1).item() == EOS_token:
			break

	return y_input.view(-1).tolist()

moves = []
while True:
	print("> ", end="")
	inp_move = input()
	if inp_move == "!q":
		break

	# inp_move = torch.tensor(list(map(int, inp_move.split())))
	print("you typed", inp_move)
	src = torch.tensor(list(map(int, inp_move.split()))).reshape(1, -1).to(device)

	print(src.size())
	print(src)
	pred = predict(model, src)

	for i in pred:
		print(TGT.vocab.itos[i], end=" ")

src = []
tgt = []
OUT = []

def test():
	cnt = 0
	# for i in test_dataloader:
	example = src[0]
	X = example.src
	y = example.tgt

	pred = predict(model,X[:,0].reshape(1,-1))
#		print("pred", pred)

	inp = X[:, 0].tolist()
	out = y[:, 0].tolist()

	print("input")
	for i in inp:
		print(SRC.vocab.itos[i], end='')

	game = []
	inp_move = ""
	for i in inp:
		if i == 3:
			break
		elif i == 5:
			game.append(inp_move)
			inp_move = ""
		elif i != 4 and i != 2:
			inp_move += SRC.vocab.itos[i]
	src.append(game)

	print()
	print()
	print("output")
	for i in out:
		print(TGT.vocab.itos[i], end='')

	game = []
	out_move = ""
	for i in out:
		if i == 3:
			break
		elif i == 5:
			game.append(out_move)
			out_move = ""
		elif i != 4 and i != 2:
			out_move += TGT.vocab.itos[i]
	OUT.append(game)

	print()
	print()
	print("pred")
	for i in pred:
		print(TGT.vocab.itos[i], end='')

	game = []
	pred_move = ""
	for i in pred:
		if i == 3:
			break
		elif i == 5:
			game.append(pred_move)
			pred_move = ""
		elif i != 4 and i != 2:
			pred_move += TGT.vocab.itos[i]
	tgt.append(game)

#test()

#print(src)
#print(tgt)
#print(OUT)


def stockfish14_scores(move_list, prediction, out):

	score = []
	board_in = chess.Board()
	for move in move_list:
		# illegal move in inital game seq
		try:
			board_in.push_san(move)
		except ValueError:
			return score
	# use the input board with initial game state

	engine = chess.engine.SimpleEngine.popen_uci("/home/harshil.p/stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")
	# Scoring ----> Mate(-0) < Mate(-1) < Cp(-50) < Cp(200) < Mate(4) < Mate(1) < MateGiven

	# using copy to have two diff boards for evaluation
	eval_board = copy.deepcopy(board_in)
	out_board = copy.deepcopy(board_in)

#    info = engine.analyse(eval_board, chess.engine.Limit(time=0.1, depth=18))
 #   score.append(info["score"].black().wdl().expectation())
    # win-draw-lose percentage from white's perspective 
	try:
		out_board.push_san(out[0])
		print("output", out[0])
	except ValueError:
		# print("Illegal move")
		score.append(float('-inf'))
	else:
		info = engine.analyse(out_board, chess.engine.Limit(time=0.1, depth=18))
		score.append(info['score'].black().wdl().expectation())

	try:
		board_in.push_san(prediction[0])
		print("pred", prediction[0])
	except ValueError:
		# print("Illegal move")
		score.append(float('-inf'))
	else:
		info = engine.analyse(board_in, chess.engine.Limit(time=0.1, depth=18))
		score.append(info['score'].black().wdl().expectation())

	result = engine.play(eval_board, chess.engine.Limit(time=0.1, depth=18))
	eval_board.push(result.move)
	info = engine.analyse(eval_board, chess.engine.Limit(time=0.1, depth=18))
	score.append(info['score'].black().wdl().expectation())

	engine.quit()
	return score, result

# pred = ['err']
# score = stockfish14_scores(src, res)
# +ve score --------> stockfish better
# if score == inf  ------> illelagl move predicted


scores = []
def get_scores():
	inf_cnt = 0
	score = [0, 0, 0]
	for i in range(len(src)):
		s, result = stockfish14_scores(src[i], tgt[i], OUT[i])
		print(f"{i}. {s}")

#		print(result)
		if s[0] == float('-inf') or s[1] == float('-inf'):
			inf_cnt += 1
		else:
			scores.append(s)

			for j in range(3):
				score[j] += s[j]
	with open("scores.txt", "wb") as f:
		pickle.dump(scores, f)

	print(scores)
	return inf_cnt, score

#inf_cnt, score = get_scores()
#print()
#print()

#for i in range(len(score)):
#	score[i] /= (10-inf_cnt)

#print(f"Incorrect moves: {inf_cnt}")
#print(score)
