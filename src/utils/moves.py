import chess
import random
import numpy as np

from .tensors import board_to_tensor, move_to_tensor

def get_neural_net_move(model, board, exploration_factor=0.1):

    legal_moves = list(board.legal_moves)
    board_copy = board.copy()  # Создаем копию доски
    move_predictions = []

    for move in legal_moves:
        board_copy.push(move)
        board_tensor = board_to_tensor(board_copy)[None, ...]  
        move_tensor = move_to_tensor(move)[None, ...]  
        prediction = model.predict([board_tensor, move_tensor])[0][0]
        move_predictions.append((move, prediction))
        board_copy.pop() 

    sorted_moves = sorted(move_predictions, key=lambda x: x[1], reverse=True)
    top_3_moves = [move[0] for move in sorted_moves[:3]]

    if (np.random.rand() < exploration_factor or board.fullmove_number == 1):
        return random.choice(top_3_moves)
    else:
        return sorted_moves[0][0]

def get_stockfish_move(board, first_move=False):
    """
    Get the best move for the given chess board using Stockfish engine.

    Parameters:
    - board (chess.Board): The chess board to get the move for.
    - first_move (bool): Flag indicating if it's the first move. If True, a random legal move will be returned.

    Returns:
    - move (chess.Move): The best move suggested by Stockfish engine.
    """
    if first_move:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

    stockfish_path = "stockfish/stockfish.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.play(board, chess.engine.Limit(depth=20, time=0.5))
    move = result.move
    engine.quit()
    return move