import pandas as pd
import chess
import chess.engine
from tensorflow.keras import models
from utils.tensors import board_to_tensor, move_to_tensor
from utils.train import train_neural_net
from utils.models import create_model
from utils.stockfish_evaluation import get_stockfish_evaluation

# Функция для загрузки и обработки исторических данных игр из CSV
def process_historical_games_data(filename, model):
    df = pd.read_csv(filename)
    training_data = []

    for index, row in df.iterrows():
        moves_str = row["Moves"]
        moves_list = moves_str.split()
        board = chess.Board()
        is_white_turn = True
        
        for i in range(1, len(moves_list), 3):  # Начинаем с 1 и увеличиваем на 3, чтобы пропустить номера ходов
            for j in range(i, i+2):  # Итерируемся по следующим двум элементам (ход белых и ход черных)
                if j < len(moves_list):  # Убеждаемся, что мы не вышли за пределы списка
                    san_move = moves_list[j]
                    try:
                        move = board.parse_san(san_move)
                        if move in board.legal_moves:
                            state = board_to_tensor(board)
                            action = move_to_tensor(move)
                            next_state = board_to_tensor(board)
                            done = board.is_game_over()
                            eval_score_before = get_stockfish_evaluation(board, is_white_turn)
                            board.push(move)  # Делаем ход
                            print(move)
                            is_white_turn = not is_white_turn
                            eval_score_after = get_stockfish_evaluation(board, is_white_turn)
                            reward = round(eval_score_after - eval_score_before, 2)  # Используем разницу оценок в качестве награды
                            training_data.append((state, action, reward, next_state, done))
                            print(eval_score_before, eval_score_after, reward)
                        else:
                            print(f"Пропускаем недопустимый ход {san_move} в строке {index}")
                    except Exception as e:
                        print(f"Пропускаем недопустимый ход {san_move} в строке {index}")
                        print(e)
                        continue

        print(len(training_data))
        model = train_neural_net(model, training_data, batch_size=128)
        training_data = []  

    return model


if __name__ == "__main__":
    historical_games_filename = "historical_games/games_clean.csv"
    model_path = "models/historical_model/model_1"

    try:
        # Загрузить или создать модель нейронной сети
        try:
            model = models.load_model(model_path)
            print(f"Загружена существующая модель: {model_path}")
        except (OSError, IOError):
            model = create_model()
            model.save(model_path)
            print(f"Создана и сохранена новая модель: {model_path}")

        if not model._is_compiled:
            model.compile(optimizer="adam", loss="mean_squared_error")

        # Загрузить исторические данные игр из CSV и обучить модель
        model = process_historical_games_data(historical_games_filename, model)

        # Сохранить обученную модель
        model.save("models/historical_model/trained_model")

    except KeyboardInterrupt:
        print("Обучение прервано")