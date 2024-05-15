import pygame
import chess
import chess.svg
import chess.engine
import os
import threading

import sys
import json
import glob
import random
import shutil
import matplotlib.pyplot as plt


import torch
import torch.optim as optim

from multiprocessing import Process, Manager, freeze_support,  Event
from pygame.locals import *
from tensorflow.keras import models
from gui.gui import draw_board, update_display
from utils.tensors import board_to_tensor, move_to_tensor
from utils.moves import get_neural_net_move
from utils.models import average_models, create_model
from utils.train import train_neural_net
from chess_net import ChessNet
from utils.moves import get_neural_net_positions

# Останній номер моделі для зберігання результатів гри
last_model_number = 0

# Шлях до файлу, де зберігаються результати гри
filename = f"models/self_learning_model/game_results.json"

# Шлях до виконуваного файлу Stockfish для використання в якості шахового двигуна
stockfish_path = f"stockfish/stockfish.exe"

# Запускаємо шаховий двигун Stockfish
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
# Шлях до збереженої моделі нейронної мережі
model_path = "C:/Users/nikit/Desktop/chess_ai/src/models/learn_evaluate/model"

# Завантажуємо модель, якщо вона існує
if os.path.exists(model_path):
    neural_net = torch.load(model_path, map_location=torch.device('cpu'))
else:   
    # Створюємо нову модель, якщо не знайдено існуючу
    neural_net = ChessNet()
    torch.save(neural_net, model_path)

# Використовуємо алгоритм оптимізації Adam для навчання моделі
optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

def get_trained_model_evaluation(board, neural_net_color):
    """
    Оцінює шахову позицію за допомогою навченої нейронної мережі.

    Параметри:
    - board: шахова дошка з поточною позицією фігур
    - neural_net_color: колір фігур, що оцінюються нейронною мережею

    Повертає:
    - оцінку позиції нейронною мережею
    """
        
    # Перетворюємо поточну шахову позицію в формат, придатний для моделі
    board_state = board_to_input(board, neural_net_color)
    
    # Перетворюємо вхідні дані у формат float і додаємо додаткове вимірювання
    board_state = board_state.float().unsqueeze(0)
    
    # Використовуємо модель для оцінки позиції
    with torch.no_grad():
        neural_net_score = neural_net(board_state)
    
    # Перетворюємо результат у тип даних float і повертаємо його
    neural_net_score = neural_net_score.item()
    return neural_net_score

def board_to_input(board, neural_net_color):
    """
    Перетворює шахову дошку в тензор, придатний для подачі на вхід нейронній мережі.

    Параметри:
    - board: шахова дошка з поточною позицією фігур
    - neural_net_color: колір фігур, що оцінюються нейронною мережею

    Повертає:
    - тензор, що представляє шахову позицію
    """
    board_state = torch.zeros(12, 8, 8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            color = int(piece.color == neural_net_color)
            piece_type = piece.piece_type - 1
            board_state[color * 6 + piece_type][i // 8][i % 8] = 1
    return board_state.unsqueeze(0)

def calculate_accuracy(neural_net_score, stockfish_score):
    """
    Розраховує точність оцінки нейронної мережі в порівнянні з оцінкою Stockfish.

    Параметри:
    - neural_net_score: оцінка позиції, надана нейронною мережею
    - stockfish_score: оцінка позиції, надана Stockfish

    Повертає:
    - розраховану точність (в процентах)
    """
    deviation = abs(neural_net_score - stockfish_score)
    return max(0, 100 - deviation * 20)

def train_neural_net2(neural_net, optimizer, board, neural_net_color, accuracies):
    """
    Навчає нейронну мережу, використовуючи дані з шахової дошки та оцінки Stockfish.

    Параметри:
    - neural_net: модель нейронної мережі
    - optimizer: оптимізатор для моделі
    - board: поточна шахова дошка з позицією
    - neural_net_color: колір, що належить нейронній мережі
    - accuracies: список для збереження точностей
    """
    # Перетворюємо шахову позицію в формат тензора для моделі
    board_state = board_to_input(board, neural_net_color)
    
    # Аналізуємо позицію за допомогою Stockfish і отримуємо її оцінку
    result = engine.analyse(board, chess.engine.Limit(depth=20, time=0.5))
    stockfish_score = result["score"].relative.score(mate_score=10000) / 100.0
    
    # Якщо оцінка Stockfish рівна нулю, виходимо з функції
    if stockfish_score == 0:
        return
    
    # Перетворюємо тензор у формат float і додаємо додатковий вимір
    board_state = board_state.float().unsqueeze(0)
    
    # Використовуємо модель для оцінки позиції
    neural_net_score = neural_net(board_state)
    
    # Обчислюємо втрати (loss) як квадрат різниці між оцінками
    loss = (neural_net_score - stockfish_score) ** 2
    
    # Очищаємо градієнти оптимізатора
    optimizer.zero_grad()
    
    # Обчислюємо градієнти та оновлюємо параметри моделі
    loss.backward()
    torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Розраховуємо точність і додаємо її в список accuracies
    accuracy = calculate_accuracy(neural_net_score.item(), stockfish_score)
    accuracies.append(accuracy)
    print(f'Stockfish score: {stockfish_score}, Neural net score: {neural_net_score.item()}, Accuracy: {accuracy}%')

def learn_position():
    """
    Вчить модель нейронної мережі на основі випадкових ігор в шахи.
    
    - Параметр games: кількість ігор для навчання.
    """
    games = 1000
    accuracies = []
    average_accuracies = []  # Список для збереження середньої точності за останню гру
    
    # Цикл по кожній грі
    for i in range(1, games + 1):
        board = chess.Board()  # Створюємо нову шахову дошку
        game_accuracies = []  # Список точностей для кожної гри
        
        # Поки гра не завершена, граємо випадкові ходи і навчаємо нейронну мережу
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
            train_neural_net2(neural_net, optimizer, board, board.turn, game_accuracies)

        # Розраховуємо середню точність за останню гру
        average_accuracy = sum(game_accuracies) / len(game_accuracies)
        average_accuracies.append(average_accuracy)
        print("\nГра була завершена. Середня точність: ", average_accuracy, "\n")
        
        print(game_accuracies)
        if i % 2 == 0 or i == games:
            plt.plot(average_accuracies)
            plt.ylim([0, 100])
            plt.xlabel('Game')
            plt.ylabel('Average Accuracy (%)')
            plt.savefig(f'models/learn_evaluate/average_accuracy_{i}.png')
            plt.close()
        
        # Якщо це не остання гра, додаємо точності поточної гри в загальний список accuracies
        if i != games:
            accuracies.extend(game_accuracies)

        # Зберігаємо нейронну мережу після кожної гри
        torch.save(neural_net, model_path)

    # Закриваємо двигун Stockfish після завершення навчання
    engine.quit()

def play_game(model, display_queue, i, replay_buffer):
    """
    Грає одну гру в шахи, використовуючи задану модель нейронної мережі.

    Параметри:
    - model: модель нейронної мережі, що використовується для гри
    - display_queue: черга для передачі візуалізації гри
    - i: індекс гри
    - replay_buffer: буфер для збереження станів гри для подальшого навчання

    Повертає:
    - model: оновлену модель після навчання
    """
    board = chess.Board()  # Створюємо нову шахову дошку
    last_move = None

    # Змінні для відстеження статистики ходів для кожного кольору
    total_moves_white = 0
    correct_moves_white = 0
    total_moves_black = 0
    correct_moves_black = 0
    neural_net_moves_white = []
    neural_net_moves_black = []
    rewards_white = []
    rewards_black = []
    last_positions_white = []
    last_positions_black = []

    # Гра поки не закінчиться
    while not board.is_game_over():
        # Визначаємо колір поточного ходу
        neural_net_color = board.turn
        
        # Отримуємо тензор поточної дошки
        state = board_to_tensor(board)
        
        # Отримуємо хід від нейронної мережі
        neural_net_move = get_neural_net_move(model, board)

        # Додаємо хід нейронної мережі в історію
        if neural_net_color == chess.WHITE:
            neural_net_moves_white.append(neural_net_move.uci())
        else:
            neural_net_moves_black.append(neural_net_move.uci())

        # Отримуємо оцінку позиції перед ходом
        eval_score_before = get_trained_model_evaluation(board, neural_net_color)
        
        # Оновлюємо дошку і робимо хід
        last_move = neural_net_move
        board.push(neural_net_move)
        changes_buffer = draw_board(board, last_move, i, neural_net_color, eval_score_before)
        display_queue.put(changes_buffer)
        
        # Отримуємо оцінку позиції після ходу
        eval_score_after = get_trained_model_evaluation(board, neural_net_color)
        
        # Перетворюємо оновлену дошку в тензор
        next_state = board_to_tensor(board)
        
        # Перевіряємо, чи гра закінчена
        done = board.is_game_over()

        # Перевірка на повторний хід і хід назад
        if neural_net_color == chess.WHITE:
            is_repeat_move = neural_net_move.uci() in neural_net_moves_white
            is_back_move = get_neural_net_positions(board, neural_net_color) in last_positions_white
            last_positions_white.append(get_neural_net_positions(board, neural_net_color))
        else:
            is_repeat_move = neural_net_move.uci() in neural_net_moves_black
            is_back_move = get_neural_net_positions(board, neural_net_color) in last_positions_black
            last_positions_black.append(get_neural_net_positions(board, neural_net_color))

        # Розраховуємо винагороду (reward) як різницю між оцінками до і після ходу
        if (is_repeat_move or is_back_move) and eval_score_after < eval_score_before:
            reward = (eval_score_after - eval_score_before) * 2
            print(f"Повтор ходу або хід назад ({'білий' if neural_net_color == chess.WHITE else 'чорний'}): {neural_net_move.uci()} Штраф був змінений: {reward}")
        else:
            reward = eval_score_after - eval_score_before
        
        # Додаємо винагороду в списки відповідного кольору
        if neural_net_color == chess.WHITE:
            rewards_white.append(reward)
        else:
            rewards_black.append(reward)

        # Перетворюємо хід у тензор
        action = move_to_tensor(neural_net_move)
        
        # Додаємо стан, дію, винагороду, наступний стан і ознаку закінчення гри в буфер повтору
        replay_buffer.append((state, action, reward, next_state, done))

        # Відстежуємо кількість і правильність ходів для кожного кольору
        if neural_net_color == chess.WHITE:
            total_moves_white += 1
            if reward > 0:
                correct_moves_white += 1
        else:
            total_moves_black += 1
            if reward > 0:
                correct_moves_black += 1

    print("Гра була завершена")

    # Додаємо підсумкову інформацію про гру для кожного кольору
    print(f"Білий: правильні ходи - {correct_moves_white}/{total_moves_white}, штрафи та нагороди - {sum(rewards_white)}")
    print(f"Чорний: правильні ходи - {correct_moves_black}/{total_moves_black}, штрафи та нагороди - {sum(rewards_black)}")

    # Тренуємо нейронну мережу
    model = train_neural_net(model, replay_buffer, 128)

    # Перевіряємо наявність файлу результатів ігор
    filename = "game_results.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            game_results = json.load(file)
    else:
        game_results = []

    # Створюємо результат гри
    game_result = {
        "Game number": len(game_results) + 1,
        "Game result": board.result(),
        "White good moves": f"{correct_moves_white / total_moves_white * 100:.2f}%",
        "Black good moves": f"{correct_moves_black / total_moves_black * 100:.2f}%",
        "White penalties and awards": f"{sum(rewards_white):.2f}",
        "Black penalties and awards": f"{sum(rewards_black):.2f}",
        "White net history": ', '.join(neural_net_moves_white),
        "Black net history": ', '.join(neural_net_moves_black)
    }

    # Додаємо результат гри до списку результатів
    game_results.append(game_result)

    # Записуємо оновлені результати ігор у файл
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(game_results, file, indent=4)

    return model

def run_game_for_color(result_queue, display_queue, i, replay_buffer, event):
    """
    Запускає гру для певного кольору (білий або чорний), використовуючи задану модель нейронної мережі.

    Параметри:
    - result_queue: черга для передачі результатів гри
    - display_queue: черга для передачі візуалізації гри
    - i: індекс гри
    - replay_buffer: буфер для збереження станів гри для подальшого навчання
    """
    try:
        # Перевіряємо наявність файлів з моделями
        files = os.listdir("models/self_learning_model")
        model_numbers = [
            int(file.split("_")[1]) for file in files if file.startswith("model_")
        ]

        # Якщо моделі знайдені, завантажуємо останню
        if model_numbers:
            last_model_number = max(model_numbers)
            model = models.load_model(
                f"models/self_learning_model/model_{last_model_number}"
            )

            # Якщо модель не скомпільована, компілюємо її
            if not model._is_compiled:
                model.compile(optimizer="adam", loss="mean_squared_error")

            print(f"Завантажена модель: {last_model_number}")
        else:
            raise IOError("Не знайдено моделей")

        # Запускаємо гру
        result = play_game(model, display_queue, i, replay_buffer)
        result_queue.put(result)
        if result_queue.qsize() == 8:
                    my_models = [result_queue.get() for _ in range(8)]

                    averaged_model = average_models(my_models)
                        
                        # Отримуємо список наявних номерів моделей
                    files = os.listdir(
                            "models/self_learning_model"
                        )
                    model_numbers = [
                            int(file.split("_")[1])
                            for file in files
                            if file.startswith("model_")
                        ]

                        # Визначаємо останній номер моделі або встановлюємо як 0
                    if model_numbers:
                            last_model_number = max(model_numbers)
                    else:
                            last_model_number = 0

                        # Визначаємо новий номер моделі і шлях для її збереження
                    new_model_number = last_model_number + 1
                    new_model_path = f"models/self_learning_model/model_{new_model_number}"

                        # Перевіряємо, чи скомпільована модель
                    if not averaged_model._is_compiled:
                            averaged_model.compile(
                                optimizer="adam", loss="mean_squared_error"
                            )
                            
                        # Тренуємо середньоважену модель
                    print("Тренування середньоваженої моделі...")
                    averaged_model = train_neural_net(
                            averaged_model, replay_buffer, 128
                        )
                        
                        # Видаляємо старі моделі
                    for old_model_path in glob.glob("models/self_learning_model/model_*"):
                            shutil.rmtree(old_model_path)

                        # Зберігаємо нову середньоважену модель
                    averaged_model.save(new_model_path)
                    print(f"Збережена модель: {new_model_number}")

                        # Зчитуємо результати ігор з файлу
                    with open(filename, "r", encoding="utf-8") as file:
                            game_results = json.load(file)

                        # Отримуємо номери ігор і суми штрафів та нагород
                    game_numbers = [result["Game number"] for result in game_results]
                    sum_penalties_awards = [float(result["Sum of penalties and awards"]) for result in game_results]

                        # Створюємо графік
                    plt.figure(figsize=(10, 6))
                    plt.plot(game_numbers, sum_penalties_awards, marker='o')
                    plt.title('Сума штрафів та нагород за гру')
                    plt.xlabel('Номер гри')
                    plt.ylabel('Сума штрафів та нагород')
                    plt.grid(True)
                    plt.savefig('models/self_learning_model/plot.png')

                    my_models = []
                    event.set()
        return model
        
    except KeyboardInterrupt:
        # Обробка клавіатурного переривання (Ctrl+C)
        print(f"Процес {os.getpid()} отримав KeyboardInterrupt. Завершення...")
        sys.exit(0)
    except: 
          return model


def play_chess_learning(num_training_cycles):
    """
    Запускає процес навчання гри в шахи.
    
    """
    freeze_support()
    
    for _ in range(num_training_cycles):
        try:
            # Використовуємо менеджер для створення спільних структур даних
            with Manager() as manager:
                replay_buffer = manager.list()  # Буфер повтору для навчання
                result_queue = manager.Queue()  # Черга для передачі результатів
                display_queue = manager.Queue()  # Черга для передачі даних для візуалізації
                processes = []  # Список процесів
                stop_flag = threading.Event()
                # Створюємо окремий потік для оновлення візуалізації

               
                # Перевіряємо наявність файлів моделей
                files = os.listdir("models/self_learning_model")
                model_numbers = [
                    int(file.split("_")[1]) for file in files if file.startswith("model_")
                ]
                
                # Якщо моделі не знайдені, створюємо нову
                if not model_numbers:
                    model = create_model()
                    new_model_path = f"models/self_learning_model/model_1"
                    model.save(new_model_path)

                event = Event()
            
                for i in range(8):
                    process = Process(
                        target=run_game_for_color,
                        args=(result_queue, display_queue, i, replay_buffer, event),
                    )
                    process.daemon = True
                    processes.append(process)
                    process.start()

                try:
                    display_thread = threading.Thread(target=update_display, args=(display_queue, True, stop_flag))
                    display_thread.daemon = True
                    display_thread.start()
                except:
                     pass
                      
                event.wait()

 
                for process in processes:
                    process.terminate()
                stop_flag.set()  

        except Exception as e:
            print(f"Error during training cycle: {e}")

        except KeyboardInterrupt:
            # Обробка клавіатурного переривання (Ctrl+C)
            for process in processes:
                process.terminate()
            pygame.quit()
            exit()

if __name__ == "__main__":
    # Вибір режиму
    print("Виберіть режим:")
    print("1. Навчання моделі гри в шахи")
    print("2. Навчання моделі оцінки шахових позицій")
    mode = input("Введіть 1 або 2: ")
    
    if mode == "1":
        play_chess_learning(1000)
    elif mode == "2":
        learn_position()