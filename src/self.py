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
import random
import shutil
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from multiprocessing import Process, Manager, freeze_support
from pygame.locals import *
from tensorflow.keras import models
from gui.gui import draw_board, update_display
from utils.tensors import board_to_tensor, move_to_tensor
from utils.moves import get_neural_net_move
from utils.models import average_models, create_model
from utils.train import train_neural_net
from chess_net import ChessNet

last_model_number = 0
num_training_cycles = 1000
filename = f"models/self_learning_model/game_results.json"
stockfish_path = f"stockfish/stockfish.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
model_path = "C:/Users/nikit/Desktop/chess_ai/src/models/learn_evaluate/model"
if os.path.exists(model_path):
    neural_net = torch.load(model_path, map_location=torch.device('cpu'))
else:   
    neural_net = ChessNet()
    torch.save(neural_net, model_path)

optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

def get_trained_model_evaluation(board, neural_net_color):
        
    # Преобразуем текущую шахматную позицию в формат, подходящий для модели
    board_state = board_to_input(board, neural_net_color)
    
    # Преобразуем входные данные в формат float и добавляем дополнительное измерение
    board_state = board_state.float().unsqueeze(0)
    
    # Используем модель для оценки позиции
    with torch.no_grad():
        neural_net_score = neural_net(board_state)
    
    # Преобразуем результат в тип данных float и возвращаем его
    neural_net_score = neural_net_score.item()
    return neural_net_score

def board_to_input(board, neural_net_color):
    board_state = torch.zeros(12, 8, 8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            color = int(piece.color == neural_net_color)
            piece_type = piece.piece_type - 1
            board_state[color * 6 + piece_type][i // 8][i % 8] = 1
    return board_state.unsqueeze(0)

def calculate_accuracy(neural_net_score, stockfish_score):
    deviation = abs(neural_net_score - stockfish_score)
    return max(0, 100 - deviation * 20)

def train_neural_net(neural_net, optimizer, board, neural_net_color, accuracies):
    board_state = board_to_input(board, neural_net_color)
    result = engine.analyse(board, chess.engine.Limit(depth=20, time=0.5))
    stockfish_score = result["score"].relative.score(mate_score=10000) / 100.0
    if stockfish_score == 0:
        return
    board_state = board_state.float().unsqueeze(0)
    neural_net_score = neural_net(board_state)
    loss = (neural_net_score - stockfish_score) ** 2
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=1.0)
    optimizer.step()
    accuracy = calculate_accuracy(neural_net_score.item(), stockfish_score)
    accuracies.append(accuracy)
    print(f'Stockfish score: {stockfish_score}, Neural net score: {neural_net_score.item()}, Accuracy: {accuracy}%')

def get_trained_model_evaluation(board, neural_net_color):
    board_state = board_to_input(board, neural_net_color)
    board_state = board_state.float().unsqueeze(0)
    # Используем модель для оценки позиции
    with torch.no_grad():
        neural_net_score = neural_net(board_state)
    neural_net_score = neural_net_score.item()
    return neural_net_score

def part2():
    games = 1000
    accuracies = []
    average_accuracies = []  # Для сохранения средней точности за последнюю игру
    for i in range(1, games + 1):
        board = chess.Board()
        game_accuracies = []  # Создаем список точностей для каждой игры
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
            train_neural_net(neural_net, optimizer, board, board.turn, game_accuracies)

        average_accuracy = sum(game_accuracies) / len(game_accuracies)  # Средняя точность за последнюю игру
        average_accuracies.append(average_accuracy)
        print("\nГра була завершена. Середня точність: ", average_accuracy, "\n")
        if i % 5 == 0 or i == games: 
            plt.plot(average_accuracies)
            plt.ylim([0, 100])
            plt.xlabel('Game')
            plt.ylabel('Average Accuracy (%)')
            plt.savefig(f'models/learn_evaluate/average_accuracy_{i}.png')
            plt.close()
        if i != games:
            accuracies.extend(game_accuracies)  # Добавляем точности текущей игры в общий список

        torch.save(neural_net, model_path)

        engine.quit()

def play_game(model, display_queue, i, replay_buffer):
    board = chess.Board()
    last_move = None

    total_moves_white = 0
    correct_moves_white = 0
    total_moves_black = 0
    correct_moves_black = 0
    neural_net_moves_white = []
    neural_net_moves_black = []
    rewards_white = []
    rewards_black = []

    while not board.is_game_over():
        neural_net_color = board.turn
        state = board_to_tensor(board)
        neural_net_move = get_neural_net_move(model, board)

        if neural_net_color == chess.WHITE:
            neural_net_moves_white.append(neural_net_move.uci())
        else:
             neural_net_moves_black.append(neural_net_move.uci())

        eval_score_before = get_trained_model_evaluation(board, neural_net_color)
        last_move = neural_net_move
        changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
        display_queue.put(changes_buffer)
        board.push(neural_net_move)
        eval_score_after = get_trained_model_evaluation(board, neural_net_color)
        next_state = board_to_tensor(board)
        done = board.is_game_over()
        reward = round(eval_score_after - eval_score_before, 2)
        print(eval_score_before, eval_score_after, reward)
        if neural_net_color == chess.WHITE:
            rewards_white.append(reward)
        else:
            rewards_black.append(reward)

        action = move_to_tensor(neural_net_move)
        replay_buffer.append((state, action, reward, next_state, done))

        if neural_net_color == chess.WHITE:
            total_moves_white += 1
            if reward > 0:
                correct_moves_white += 1
        else:
            total_moves_black += 1
            if reward > 0:
                correct_moves_black += 1

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
        "Sum of penalties and awards": f"{sum(rewards_white + rewards_black):.2f}",
        "WHITE Good moves": f"{correct_moves_white / total_moves_white * 100:.2f}%",
        "BLACK Good moves": f"{correct_moves_black / total_moves_black * 100:.2f}%",
        "Net WHITE history": ', '.join(neural_net_moves_white),
        "NET BLACK history": ', '.join(neural_net_moves_black),
        "WHITE Penalties and awards": ', '.join(map(str, rewards_white)),
        "BLACK Penalties and awards": ', '.join(map(str, rewards_black))
    }
    game_results.append(game_result)

    # Write updated game results back to file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(game_results, file, indent=4)

    return model

def run_game_for_color(result_queue, display_queue, i, replay_buffer):

    try:
        files = os.listdir("models/self_learning_model")
        model_numbers = [
            int(file.split("_")[1]) for file in files if file.startswith("model_")
        ]

        if model_numbers:
            last_model_number = max(model_numbers)
            model = models.load_model(
                f"models/self_learning_model/model_{last_model_number}"
            )

            if not model._is_compiled:
                model.compile(optimizer="adam", loss="mean_squared_error")

            print(f"Loaded model: {last_model_number}")
        else:
            raise IOError("No models was found")

        result = play_game(model, display_queue, i, replay_buffer)
        result_queue.put(result)
        return
    except KeyboardInterrupt:
        print(f"Process {os.getpid()} received KeyboardInterrupt. Terminating...")
        sys.exit(0)

def part1():
    try:
        freeze_support()
        with Manager() as manager:
            replay_buffer = manager.list()
            result_queue = manager.Queue()
            display_queue = manager.Queue()
            processes = []

            display_thread = threading.Thread(target=update_display, args=(display_queue, True,))
            display_thread.daemon = True
            display_thread.start()

            files = os.listdir("models/self_learning_model")
            model_numbers = [
                int(file.split("_")[1]) for file in files if file.startswith("model_")
            ]
            if not model_numbers:
                model = create_model()
                new_model_path = f"models/self_learning_model/model_1"
                model.save(new_model_path)

            for _ in range(num_training_cycles):
                for i in range(8):
                    process = Process(
                        target=run_game_for_color,
                        args=(result_queue, display_queue, i, replay_buffer),
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
                            "models/self_learning_model"
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
                        new_model_path = f"models/self_learning_model/model_{new_model_number}"

                        if not averaged_model._is_compiled:
                            averaged_model.compile(
                                optimizer="adam", loss="mean_squared_error"
                            )
                            
                        print("Training averaged model...")
                        averaged_model = train_neural_net(
                            averaged_model, replay_buffer, 64
                        )
                        
                        for old_model_path in glob.glob("models/self_learning_model/model_*"):
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
                        plt.savefig('models/self_learning_model/plot.png')

                processes = []
                my_models = []
                
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        pygame.quit()
        exit()

if __name__ == "__main__":
    print("Выберите режим:")
    print("1. Обучение модели игры в шахматы")
    print("2. Обучение модели оценки шахматных позиций")
    mode = input("Введите 1 или 2: ")
    if mode == "1":
        part1()
    elif mode == "2":
        part2()