import chess

def get_stockfish_evaluation(board, neural_net_color):
    """
    Calculates the evaluation score of the given chess board using Stockfish engine.

    Args:
        board (chess.Board): The chess board to evaluate.
        neural_net_color (bool): The color of the neural network player.

    Returns:
        float: The evaluation score of the board. Positive score indicates an advantage for the neural network player,
               while negative score indicates an advantage for the opponent.
    """
    stockfish_path = "stockfish/stockfish.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.analyse(board, chess.engine.Limit(depth=20, time=0.5))
    engine.quit()

    eval_score = result["score"].relative.score(mate_score=10000) / 100.0

    if board.turn == neural_net_color:
        return eval_score
    else:
        return -eval_score
    