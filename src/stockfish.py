import pygame
import chess
import chess.svg
import chess.engine
import os
import threading
import signal
import sys
import json
import glob
import shutil
import matplotlib.pyplot as plt

from multiprocessing import Process, Manager, freeze_support
from pygame.locals import *

from tensorflow.keras import models
from gui.gui import draw_board, update_display
from utils.tensors import board_to_tensor, move_to_tensor
from utils.moves import get_neural_net_move, get_stockfish_move
from utils.models import average_models, create_model
from utils.train import train_neural_net
from utils.stockfish_evaluation import get_stockfish_evaluation
from utils.moves import get_neural_net_positions

# Встановлюємо початковий номер моделі
last_model_number = 0

# Встановлюємо кількість навчальних циклів
num_training_cycles = 1000

# Шлях до файлу з результатами ігор
filename = f"models/reinforcement_learning_stockfish_model/game_results.json"

def play_game(model, neural_net_color, display_queue, i, replay_buffer):
    """
    Функція для гри в шахи між нейронною мережею та Stockfish.

    Параметри:
    - model: нейронна мережа для гри в шахи.
    - neural_net_color: колір сторони нейронної мережі (білий або чорний).
    - display_queue: черга для відображення змін дошки.
    - i: номер гри.
    - replay_buffer: буфер для навчання нейронної мережі.
    """
    # Створюємо дошку для гри
    board = chess.Board()
    last_move = None

    total_moves = 0  # Загальна кількість ходів
    correct_moves = 0  # Кількість правильних ходів
    neural_net_moves = []  # Історія ходів нейронної мережі
    stockfish_moves = []  # Історія ходів Stockfish
    rewards = []  # Нагороди за правильні ходи
    last_positions = []

    first_move = True  # Прапор для перевірки першого ходу

    # Гра триває, доки не завершиться
    while not board.is_game_over():
        state = board_to_tensor(board)
        neural_net_move = None
        stockfish_move = None
        eval_score_before = get_stockfish_evaluation(board, neural_net_color)

        # Якщо черга нейронної мережі
        if board.turn == neural_net_color:
            # Отримуємо хід від нейронної мережі
            neural_net_move = get_neural_net_move(model, board)
            neural_net_moves.append(neural_net_move.uci())
            board.push(neural_net_move)
            eval_score_after = get_stockfish_evaluation(board, neural_net_color)
            last_move = neural_net_move
            changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
            display_queue.put(changes_buffer)
            last_positions.append(get_neural_net_positions(board, neural_net_color))
        else:
            # Отримуємо хід від Stockfish
            stockfish_move = get_stockfish_move(board, first_move)
            first_move = False
            board.push(stockfish_move)
            last_move = stockfish_move
            eval_score_after = get_stockfish_evaluation(board, neural_net_color)
            changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
            display_queue.put(changes_buffer)
            stockfish_moves.append(stockfish_move.uci())
            

        next_state = board_to_tensor(board)
        done = board.is_game_over()
        uci_move = neural_net_move.uci() if neural_net_move else None
        is_repeat_move = uci_move in neural_net_moves if uci_move else False
        is_back_move = get_neural_net_positions(board, neural_net_color) in last_positions
        if (is_repeat_move or is_back_move) and eval_score_after < eval_score_before:
            reward = (eval_score_after - eval_score_before) * 2
            print(is_repeat_move, is_back_move)
            print("Повтор ходу або хід назад: ", uci_move, "Штраф був змінений: ", reward)
        else:
            reward = eval_score_after - eval_score_before
        
        print(reward)
        # Перетворюємо хід у тензор
        action = move_to_tensor(last_move)

        # Додаємо епізод у буфер повторення для навчання
        replay_buffer.append((state, action, reward, next_state, done))

        if reward > 0:
            correct_moves += 1

        total_moves += 1

    print("Гра завершена")

    # Тренуємо нейронну мережу
    model = train_neural_net(model, replay_buffer, 128)

    # Перевіряємо наявність файлу результатів ігор
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            game_results = json.load(file)
    else:
        game_results = []

    # Створюємо результат гри
    game_result = {
        "Game number": len(game_results) + 1,
        "Game result": board.result(),
        "Side": 'White' if neural_net_color == chess.WHITE else 'Black',
        "Sum of penalties and awards": f"{sum(rewards):.2f}",
        "Good moves": f"{correct_moves / total_moves * 100:.2f}%",
        "Net history": ', '.join(neural_net_moves),
        "Stockfish history": ', '.join(stockfish_moves),
        "Penalties and awards": ', '.join(map(str, rewards))
    }

    # Додаємо результат гри до списку результатів
    game_results.append(game_result)

    # Записуємо оновлені результати ігор у файл
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(game_results, file, indent=4)

    return model

def run_game_for_color(color, result_queue, display_queue, i, replay_buffer):
    """
    Функція для запуску гри для певного кольору (білий або чорний) з нейронною мережею та Stockfish.

    Параметри:
    - color: колір сторони гри (білий або чорний).
    - result_queue: черга для результатів гри.
    - display_queue: черга для відображення змін дошки.
    - i: номер гри.
    - replay_buffer: буфер для навчання нейронної мережі.
    """
    try:
        # Знаходимо всі моделі у теці
        files = os.listdir("models/reinforcement_learning_stockfish_model")
        model_numbers = [
            int(file.split("_")[1]) for file in files if file.startswith("model_")
        ]

        # Якщо моделі існують, завантажуємо останню
        if model_numbers:
            last_model_number = max(model_numbers)
            model = models.load_model(
                f"models/reinforcement_learning_stockfish_model/model_{last_model_number}"
            )

            # Перевіряємо, чи модель скомпільована
            if not model._is_compiled:
                model.compile(optimizer="adam", loss="mean_squared_error")

            print(f"Loaded model: {last_model_number}")
        else:
            raise IOError("No models found")

        # Граємо гру
        result = play_game(model, color, display_queue, i, replay_buffer)
        # Додаємо результат до черги результатів
        result_queue.put(result)
        return
    except KeyboardInterrupt:
        print(f"Process {os.getpid()} received KeyboardInterrupt. Terminating...")
        sys.exit(0)

if __name__ == "__main__":
    try:
        freeze_support()

        with Manager() as manager:
            # Створюємо черги та буфери для управління процесами
            replay_buffer = manager.list()
            result_queue = manager.Queue()
            display_queue = manager.Queue()
            processes = []
            stop_flag = threading.Event()
            # Запускаємо потік для оновлення дисплею
            display_thread = threading.Thread(target=update_display, args=(display_queue, False, stop_flag))
            display_thread.daemon = True
            display_thread.start()

            # Знаходимо всі моделі у папці
            files = os.listdir("models/reinforcement_learning_stockfish_model")
            model_numbers = [
                int(file.split("_")[1]) for file in files if file.startswith("model_")
            ]
            # Якщо немає моделей, створюємо нову
            if not model_numbers:
                model = create_model()
                new_model_path = f"models/reinforcement_learning_stockfish_model/model_1"
                model.save(new_model_path)

            # Цикл навчання
            for _ in range(num_training_cycles):
                # Запускаємо процеси для гри за обома кольорами
                for i in range(8):
                    color = chess.BLACK
                    process = Process(
                        target=run_game_for_color,
                        args=(color, result_queue, display_queue, i, replay_buffer),
                    )
                    process.daemon = True
                    processes.append(process)
                    process.start()

                    def signal_handler(sig, frame):
                        for process in processes:
                            process.terminate()
                        sys.exit(1)

                signal.signal(signal.SIGINT, signal_handler)

                # Чекаємо завершення процесів
                for process in processes:
                    process.join()

                # Якщо є результати в черзі
                if not result_queue.empty():
                    my_models = [result_queue.get() for _ in processes]

                    # Якщо отримано достатньо моделей
                    if len(my_models) >= 8:
                        averaged_model = average_models(my_models)
                        files = os.listdir(
                            "models/reinforcement_learning_stockfish_model"
                        )
                        model_numbers = [
                            int(file.split("_")[1])
                            for file in files
                            if file.startswith("model_")
                        ]

                        # Знаходимо останній номер моделі
                        if model_numbers:
                            last_model_number = max(model_numbers)
                        else:
                            last_model_number = 0

                        # Визначаємо номер нової моделі
                        new_model_number = last_model_number + 1
                        new_model_path = f"models/reinforcement_learning_stockfish_model/model_{new_model_number}"

                        # Перевіряємо, чи модель скомпільована
                        if not averaged_model._is_compiled:
                            averaged_model.compile(
                                optimizer="adam", loss="mean_squared_error"
                            )

                        print("Training averaged model...")
                        # Тренуємо нейронну мережу
                        averaged_model = train_neural_net(
                            averaged_model, replay_buffer, 128
                        )

                        # Видаляємо старі моделі
                        for old_model_path in glob.glob("models/reinforcement_learning_stockfish_model/model_*"):
                            shutil.rmtree(old_model_path)

                        # Зберігаємо нову модель
                        averaged_model.save(new_model_path)
                        print(f"Saved model: {new_model_number}")

                        # Відкриваємо файл з результатами ігор
                        with open(filename, "r", encoding="utf-8") as file:
                            game_results = json.load(file)

                        # Отримуємо дані про ігри
                        game_numbers = [result["Game number"] for result in game_results]
                        sum_penalties_awards = [float(result["Sum of penalties and awards"]) for result in game_results]

                        # Створюємо графік
                        plt.figure(figsize=(10, 6))
                        plt.plot(game_numbers, sum_penalties_awards, marker='o')
                        plt.title('Sum of Penalties and Awards per Game')
                        plt.xlabel('Game Number')
                        plt.ylabel('Sum of Penalties and Awards')
                        plt.grid(True)
                        plt.savefig('models/reinforcement_learning_stockfish_model/plot.png')

                # Очищаємо списки процесів та моделей
                processes = []
                my_models = []
                results = []

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        pygame.quit()
        exit()
