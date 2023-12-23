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

last_model_number = 0
num_training_cycles = 1000
filename = f"models/reinforcement_learning_stockfish_model/game_results.json"

def play_game(model, neural_net_color, display_queue, i, replay_buffer):

    board = chess.Board()
    last_move = None

    total_moves = 0
    correct_moves = 0
    neural_net_moves = []
    stockfish_moves = []
    rewards = []

    first_move = True

    while not board.is_game_over():
        state = board_to_tensor(board)
        neural_net_move = None
        stockfish_move = None
        eval_score_before = get_stockfish_evaluation(board, neural_net_color)

        if board.turn == neural_net_color:
            neural_net_move = get_neural_net_move(model, board)
            neural_net_moves.append(neural_net_move.uci())
            board.push(neural_net_move)
            eval_score_after = get_stockfish_evaluation(board, neural_net_color)
            last_move = neural_net_move
            changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
            display_queue.put(changes_buffer)
        else:
            stockfish_move = get_stockfish_move(board, first_move)
            first_move = False
            board.push(stockfish_move)
            last_move = stockfish_move
            changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
            display_queue.put(changes_buffer)
            stockfish_moves.append(stockfish_move.uci())
            continue

        next_state = board_to_tensor(board)
        done = board.is_game_over()
        reward = round(eval_score_after - eval_score_before, 2)
        rewards.append(reward)

        action = move_to_tensor(neural_net_move)

        replay_buffer.append((state, action, reward, next_state, done))

        if reward > 0:
            correct_moves += 1

        total_moves += 1

    print("Game was finished")

    model = train_neural_net(model, replay_buffer, 64)

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            game_results = json.load(file)
    else:
        game_results = []

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
    game_results.append(game_result)

    # Записываем обновленные результаты игр обратно в файл
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(game_results, file, indent=4)

    return model

def run_game_for_color(color, result_queue, display_queue, i, replay_buffer):

    try:
        files = os.listdir("models/reinforcement_learning_stockfish_model")
        model_numbers = [
            int(file.split("_")[1]) for file in files if file.startswith("model_")
        ]

        if model_numbers:
            last_model_number = max(model_numbers)
            model = models.load_model(
                f"models/reinforcement_learning_stockfish_model/model_{last_model_number}"
            )

            if not model._is_compiled:
                model.compile(optimizer="adam", loss="mean_squared_error")

            print(f"Loaded model: {last_model_number}")
        else:
            raise IOError("No models was found")

        result = play_game(model, color, display_queue, i, replay_buffer)
        result_queue.put(result)
        return
    except KeyboardInterrupt:
        print(f"Process {os.getpid()} received KeyboardInterrupt. Terminating...")
        sys.exit(0)


if __name__ == "__main__":
    try:
        freeze_support()

        with Manager() as manager:
            replay_buffer = manager.list()
            result_queue = manager.Queue()
            display_queue = manager.Queue()
            processes = []

            display_thread = threading.Thread(target=update_display, args=(display_queue, False, ))
            display_thread.daemon = True
            display_thread.start()

            files = os.listdir("models/reinforcement_learning_stockfish_model")
            model_numbers = [
                int(file.split("_")[1]) for file in files if file.startswith("model_")
            ]
            if not model_numbers:
                model = create_model()
                new_model_path = f"models/reinforcement_learning_stockfish_model/model_1"
                model.save(new_model_path)

            for _ in range(num_training_cycles):
                for i in range(8):
                    color = chess.WHITE if i < 4 else chess.BLACK
                    process = Process(
                        target=run_game_for_color,
                        args=(color, result_queue, display_queue, i, replay_buffer),
                    )
                    process.daemon = True
                    processes.append(process)
                    process.start()

                def signal_handler(sig, frame):
                    process.terminate()
                    sys.exit(1)

                signal.signal(signal.SIGINT, signal_handler)

                for process in processes:
                    process.join()

                if not result_queue.empty():
                    my_models = [result_queue.get() for _ in processes]

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

                        if model_numbers:
                            last_model_number = max(model_numbers)
                        else:
                            last_model_number = 0

                        new_model_number = last_model_number + 1
                        new_model_path = f"models/reinforcement_learning_stockfish_model/model_{new_model_number}"

                        if not averaged_model._is_compiled:
                            averaged_model.compile(
                                optimizer="adam", loss="mean_squared_error"
                            )
                            
                        print("Training averaged model...")
                        averaged_model = train_neural_net(
                            averaged_model, replay_buffer, 64
                        )
                        
                        for old_model_path in glob.glob("models/reinforcement_learning_stockfish_model/model_*"):
                                shutil.rmtree(old_model_path)

                        averaged_model.save(new_model_path)
                        print(f"Saved model: {new_model_number}")

                        with open(filename, "r", encoding="utf-8") as file:
                            game_results = json.load(file)

                        game_numbers = [result["Game number"] for result in game_results]
                        sum_penalties_awards = [float(result["Sum of penalties and awards"]) for result in game_results]

                        # Создаем график
                        plt.figure(figsize=(10, 6))
                        plt.plot(game_numbers, sum_penalties_awards, marker='o')
                        plt.title('Sum of Penalties and Awards per Game')
                        plt.xlabel('Game Number')
                        plt.ylabel('Sum of Penalties and Awards')
                        plt.grid(True)
                        plt.savefig('models/reinforcement_learning_stockfish_model/plot.png')

                processes = []
                my_models = []
                results = []
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        pygame.quit()
        exit()
