import chess
import numpy as np

def board_to_tensor(board):
    """
    Converts a chess board to a tensor representation.

    Args:
        board (chess.Board): The chess board object.

    Returns:
        numpy.ndarray: The tensor representation of the board.
    """
    tensor = np.zeros((8, 8, 13), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            tensor[square // 8, square % 8, piece.piece_type] = 1.0

    tensor[:, :, :12] /= 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor


def move_to_tensor(move):
    """
    Converts a chess move to a tensor representation.

    Parameters:
    move (chess.Move): The chess move to convert.

    Returns:
    np.ndarray: The tensor representation of the move.
    """
    from_square = move.from_square
    to_square = move.to_square

    move_tensor = np.zeros((8, 8, 13), dtype=np.float32)
    move_tensor[from_square // 8, from_square % 8, 12] = 1.0
    move_tensor[to_square // 8, to_square % 8, 12] = 1.0

    return move_tensor