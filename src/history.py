import pandas as pd
import chess
import chess.engine
from tensorflow.keras import models
from utils.tensors import board_to_tensor, move_to_tensor
from utils.train import train_neural_net
from utils.models import create_model
from utils.stockfish_evaluation import get_stockfish_evaluation

# Функція для завантаження та обробки історичних даних ігор з CSV
def process_historical_games_data(filename, model):
    df = pd.read_csv(filename)  # Завантажуємо дані з файлу CSV у DataFrame
    training_data = []  # Список для зберігання даних для навчання моделі

    # Проходимося по кожному рядку DataFrame
    for index, row in df.iterrows():
        moves_str = row["Moves"]  # Отримуємо рядок з ходами гри
        moves_list = moves_str.split()  # Розбиваємо рядок на список ходів
        board = chess.Board()  # Створюємо нову шахову дошку
        is_white_turn = True  # Змінна, що вказує на чергу ходу білих

        # Проходимося по ходах у грі
        for i in range(1, len(moves_list), 3):  # Починаємо з 1 і збільшуємо на 3, щоб пропустити номери ходів
            for j in range(i, i + 2):  # Ітеруємося по наступних двох елементах (хід білих і хід чорних)
                if j < len(moves_list):  # Перевіряємо, що не вийшли за межі списку
                    san_move = moves_list[j]  # Отримуємо наступний хід у форматі SAN
                    try:
                        move = board.parse_san(san_move)  # Парсуємо хід у форматі SAN
                        if move in board.legal_moves:  # Перевіряємо, чи є хід допустимим
                            state = board_to_tensor(board)  # Отримуємо поточний стан дошки
                            action = move_to_tensor(move)  # Отримуємо дію у форматі тензора
                            next_state = board_to_tensor(board)  # Отримуємо наступний стан дошки
                            done = board.is_game_over()  # Перевіряємо, чи гра закінчена
                            eval_score_before = get_stockfish_evaluation(board, is_white_turn)  # Оцінка до ходу
                            board.push(move)  # Робимо хід
                            print(move)  # Виводимо хід
                            is_white_turn = not is_white_turn  # Змінюємо чергу ходу
                            eval_score_after = get_stockfish_evaluation(board, is_white_turn)  # Оцінка після ходу
                            reward = round(eval_score_after - eval_score_before, 2)  # Використовуємо різницю оцінок як винагороду
                            training_data.append((state, action, reward, next_state, done))  # Додаємо дані для навчання
                            print(eval_score_before, eval_score_after, reward)
                        else:
                            print(f"Пропускаємо недопустимий хід {san_move} у рядку {index}")
                    except Exception as e:
                        print(f"Пропускаємо недопустимий хід {san_move} у рядку {index}")
                        print(e)
                        continue

        print(len(training_data))  # Виводимо кількість зібраних даних для навчання
        model = train_neural_net(model, training_data, batch_size=128)  # Навчаємо модель на зібраних даних
        training_data = []  # Очищаємо список з даними після навчання
    return model  # Повертаємо оновлену модель

if __name__ == "__main__":
    historical_games_filename = "historical_games/games_clean.csv"  # Файл з історичними даними ігор
    model_path = "models/historical_model/model_1"  # Шлях до моделі нейронної мережі

    try:
        # Завантажити або створити модель нейронної мережі
        try:
            model = models.load_model(model_path)  # Спробуємо завантажити існуючу модель
            print(f"Завантажено існуючу модель: {model_path}")
        except (OSError, IOError):
            model = create_model()  # Якщо модель не знайдено, створимо нову
            model.save(model_path)  # Збережемо нову модель
            print(f"Створено і збережено нову модель: {model_path}")

        if not model._is_compiled:  # Якщо модель ще не скомпільована
            model.compile(optimizer="adam", loss="mean_squared_error")

        # Завантажити історичні дані ігор з CSV та навчити модель
        model = process_historical_games_data(historical_games_filename, model)

        # Зберегти навчену модель
        model.save("models/historical_model/trained_model")

    except KeyboardInterrupt:
        print("Навчання перервано")
