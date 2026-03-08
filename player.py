import chess
import random
from peft import PeftModel
import torch
from typing import Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player

class TransformerPlayer(Player):

    def __init__(
        self,
        name: str,
        hf_repo: str = "grapju/trained_model_chess",
        model_id: str = "Qwen/Qwen2.5-0.5B",
        enable_cache: bool = True,
        moves_per_candidate_group: int = 8,
    ):
        super().__init__(name)

        # LoRA adapter repository
        self.hf_repo = hf_repo

        # model used during training (see ipynb for training)
        self.model_id = model_id

        self.enable_cache = enable_cache
        self.moves_per_candidate_group = moves_per_candidate_group

        # cache earlier seen positions
        self.cache: Dict[str, str] = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading model from {self.hf_repo}")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
      
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # if running on CPU move model explicitly
        if self.device == "cpu":
            model.to(self.device)

        # attach LoRA adapter
        self.model = PeftModel.from_pretrained(model, self.hf_repo)

        # inference only
        self.model.eval()

        # store model device
        self.model_device = next(self.model.parameters()).device

        # values see source below
        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.2,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }

    def _build_prompt(self, fen: str) -> str:
        # prompt
        return f"FEN: {fen}\nMOVE: "

    def _find_checkmate(self, board: chess.Board) -> Optional[str]:
        # check if any legal move results in checkmate
        for move in board.legal_moves:
            board.push(move)

            if board.is_checkmate():
                board.pop()
                return move.uci()

            board.pop()

        return None

    def _select_candidate_moves(self, board: chess.Board, legal_moves: list[str]) -> list[str]:
      
        # if not a lot of moves, then keep all
        if len(legal_moves) <= 20:
            return legal_moves

        promotions = []
        captures = []
        checks = []
        other_moves = []

        # loop through legal moves
        for move_uci in legal_moves:

            move = chess.Move.from_uci(move_uci)

            # give high weight to promotion
            if move.promotion is not None:
                promotions.append(move_uci)
                continue

            # capture moves
            if board.is_capture(move):
                captures.append(move_uci)
                continue

            # check moves
            board.push(move)

            if board.is_check():
                checks.append(move_uci)
            else:
                other_moves.append(move_uci)

            board.pop()
        
        candidate_moves = promotions + captures + checks

        # fill remaining moves with other moves
        remaining = 20 - len(candidate_moves)

        if remaining > 0:
            candidate_moves += other_moves[:remaining]

        return candidate_moves[:20]

    def _move_bonus(self, board: chess.Board, move: chess.Move) -> float:
        # gives bonus for specific moves
        bonus = 0.0

        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)

        # add bonus for promotion
        if move.promotion is not None:
            # higher bonus if pawn becomes queen
            if move.promotion == chess.QUEEN:
                bonus += 0.4
            else:
                bonus += 0.2

        # add bonus for capturing a piece
        if captured_piece and moving_piece:

            captured_value = self.piece_values.get(captured_piece.piece_type, 0)
            moving_value = self.piece_values.get(moving_piece.piece_type, 0)

            # prefer capturing valuable pieces
            bonus += 0.03 * captured_value

            # extra bonus for good trades
            bonus += 0.02 * max(captured_value - moving_value, 0)

        # temporarily apply move
        board.push(move)

        # add bonus for check
        if board.is_check():
            bonus += 0.06

        # avoid stalemate
        if board.is_stalemate():
            bonus -= 1.5

        # check if opponent move captures valuable piece
        for opponent_move in board.legal_moves:

            if board.is_capture(opponent_move):

                captured = board.piece_at(opponent_move.to_square)

                if captured is not None:

                    captured_value = self.piece_values.get(captured.piece_type, 0)

                    # penalize losing valuable pieces
                    if captured_value >= 3:
                        bonus -= 0.08 * captured_value

        board.pop()

        return bonus

    def _score_candidate_group(self, board: chess.Board, prompt: str, moves: list[str]) -> list[float]:
        # candidate texts
        texts = [prompt + move for move in moves]

        # tokenize
        inputs = self.tokenizer(texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,)

        # move tensors to model device
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}

        # length of prompt tokens
        prompt_len = len(self.tokenizer(prompt)["input_ids"])

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits[:, :-1, :]
            input_ids = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]

            # convert logits to log prob
            log_probs = torch.log_softmax(logits, dim=-1)

            # extract log proability of each token
            token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # mask that shows which tokens belong to move
            mask = attention_mask.bool()

            # ignore prompt tokens
            if prompt_len > 1:
                mask[:, :prompt_len - 1] = False

            # sum log probability over move tokens + count them
            summed = (token_log_probs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)

            # average log prob per move token
            scores = (summed / counts).detach().cpu().tolist()

        # add bonus
        final_scores = []

        for move_uci, score in zip(moves, scores):

            move = chess.Move.from_uci(move_uci)

            final_scores.append(score + self._move_bonus(board, move))

        return final_scores

    def _score_moves(self, fen: str, candidate_moves: list[str]) -> str:
        board = chess.Board(fen)
        prompt = self._build_prompt(fen)

        best_move = candidate_moves[0]
        best_score = float("-inf")

        # loop through candidate moves per group
        for i in range(0, len(candidate_moves), self.moves_per_candidate_group):

            candidate_group = candidate_moves[i:i + self.moves_per_candidate_group]

            scores = self._score_candidate_group(board, prompt, candidate_group)
          
            for move_uci, score in zip(candidate_group, scores):

                if score > best_score:
                    best_score = score
                    best_move = move_uci

        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        # create board from fen
        try:
            board = chess.Board(fen)

        except Exception:
            return None

        legal_moves = [move.uci() for move in board.legal_moves]

        # no legal moves
        if not legal_moves:
            return None

        # if only one legal move use that
        if len(legal_moves) == 1:
            return legal_moves[0]

        # check for mate in one move
        mate_move = self._find_checkmate(board)

        if mate_move:
            if self.enable_cache:
                self.cache[fen] = mate_move

            return mate_move

        # check cache for earlier seen positions
        if self.enable_cache and fen in self.cache:
            return self.cache[fen]

        # reduce number of possible moves
        candidate_moves = self._select_candidate_moves(board, legal_moves)

        try:
            best_move = self._score_moves(fen, candidate_moves)

        except Exception:

            # random move
            best_move = random.choice(legal_moves)

        # store result in cache
        if self.enable_cache:
            self.cache[fen] = best_move

        return best_move

# source: https://en.wikipedia.org/wiki/Chess_piece_relative_value
