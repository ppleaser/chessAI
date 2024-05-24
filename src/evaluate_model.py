import chess
import chess.engine
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from utils.tensors import board_to_tensor, move_to_tensor
from utils.stockfish_evaluation import get_stockfish_evaluation
from tkinter import Tk, filedialog, Button, Label, Entry, StringVar, DISABLED, NORMAL
import os
import threading

from utils.moves import get_neural_net_move, get_stockfish_move
from utils.tensors import board_to_tensor, move_to_tensor

def get_neural_net_move(model, board):
    legal_moves = list(board.legal_moves)
    board_copy = board.copy()
    move_predictions = []

    for move in legal_moves:
        board_copy.push(move)
        board_tensor = board_to_tensor(board_copy)[None, ...]
        move_tensor = move_to_tensor(move)[None, ...]
        prediction = model.predict([board_tensor, move_tensor])[0][0]
        move_predictions.append((move, prediction))
        board_copy.pop()

    sorted_moves = sorted(move_predictions, key=lambda x: x[1], reverse=True)
    return sorted_moves[0][0]

def evaluate_model(model, model_path, game_num, save_dir, stop_event, results, test_num):
    board = chess.Board()
    first_move = True
    print(f"Тест №{test_num}")

    move_counter = 0

    while not board.is_game_over() and move_counter < 50 and not stop_event.is_set():
        eval_score_before_stockfish = get_stockfish_evaluation(board, chess.WHITE)
        eval_score_after_model = -100
        neural_net_move = get_neural_net_move(model, board)
        board.push(neural_net_move)
        eval_score_after_model = get_stockfish_evaluation(board, chess.WHITE)
        if (eval_score_after_model) < -10:
            eval_score_after_model = -10
        board.pop()
        model_improvement = eval_score_after_model - eval_score_before_stockfish
        results['model_scores'].append((move_counter, model_improvement))

        stockfish_move = get_stockfish_move(board, first_move)
        first_move = False
        board.push(stockfish_move)

        eval_score_after_stockfish = get_stockfish_evaluation(board, chess.WHITE)

        stockfish_improvement = eval_score_after_stockfish - eval_score_before_stockfish
        results['stockfish_scores'].append((move_counter, stockfish_improvement))

        stockfish_move = get_stockfish_move(board, first_move)
        board.push(stockfish_move)

        print(f"Оцінка від Stockfish до: {eval_score_before_stockfish}")
        print(f"Оцінка від Stockfish після ходу моделі: {eval_score_after_model}")
        print(f"Оцінка Stockfish після свого ходу: {eval_score_after_stockfish}")

        move_counter += 1
        update_plot(results, model_path, test_num)

    mean_model_improvement = np.mean([score for _, score in results['model_scores']])
    mean_stockfish_improvement = np.mean([score for _, score in results['stockfish_scores']])
    correlation = np.corrcoef(
        [score for _, score in results['model_scores']],
        [score for _, score in results['stockfish_scores']]
    )[0, 1] if len(results['model_scores']) > 1 else 0

    print(f"Середнє покращення від ходів моделі: {mean_model_improvement}")
    print(f"Середнє покращення від ходів Stockfish: {mean_stockfish_improvement}")
    print(f"Кореляція між покращеннями моделі та Stockfish: {correlation}")
    print(f"Гра завершена. Результат: {board.result()}")

    save_plot(model_path, test_num, save_dir, game_num)

def update_plot(results, model_path, test_num):
    plt.clf()
    max_moves = max(results['model_scores'], key=lambda x: x[0])[0] + 1

    average_model_scores = [0] * max_moves
    average_stockfish_scores = [0] * max_moves

    for i in range(max_moves):
        model_scores_at_i = [score for move, score in results['model_scores'] if move == i]
        stockfish_scores_at_i = [score for move, score in results['stockfish_scores'] if move == i]
        if model_scores_at_i:
            average_model_scores[i] = np.mean(model_scores_at_i)
        if stockfish_scores_at_i:
            average_stockfish_scores[i] = np.mean(stockfish_scores_at_i)

    plt.plot(range(max_moves), average_model_scores, label='Середнє покращення моделі')
    plt.plot(range(max_moves), average_stockfish_scores, label='Середнє покращення Stockfish')
    plt.xlabel('Ходи')
    plt.ylabel('Оцінки')
    plt.legend()
    plt.ylim(-10, max(max(average_model_scores), max(average_stockfish_scores)) + 1)
    plt.title(f"Тест №{test_num} - Модель: {os.path.basename(model_path)}")

    mean_model_improvement = np.mean(average_model_scores)
    mean_stockfish_improvement = np.mean(average_stockfish_scores)
    correlation = np.corrcoef(
        [score for _, score in results['model_scores']],
        [score for _, score in results['stockfish_scores']]
    )[0, 1] if len(results['model_scores']) > 1 else 0

    plt.figtext(0.15, 0.85, f"Середнє покращення моделі: {mean_model_improvement:.2f}")
    plt.figtext(0.15, 0.80, f"Середнє покращення Stockfish: {mean_stockfish_improvement:.2f}")
    plt.figtext(0.15, 0.75, f"Кореляція: {correlation:.2f}")

    plt.draw()
    plt.pause(0.01)

def save_plot(model_path, test_num, save_dir, game_num):
    plt.savefig(os.path.join(save_dir, f"game_{test_num}_{game_num}_{os.path.basename(model_path)}.png"))

def load_model():
    file_path = filedialog.askdirectory()
    model = models.load_model(file_path)

    if not model._is_compiled:
        model.compile(optimizer="adam", loss="mean_squared_error")
    
    return model, file_path

def select_save_directory():
    return filedialog.askdirectory()

def evaluate_multiple_games(model, model_path, save_dir, num_games, stop_event, test_num):
    results = {'model_scores': [], 'stockfish_scores': []}
    for game_number in range(1, num_games + 1):
        if stop_event.is_set():
            break
        evaluate_model(model, model_path, game_number, save_dir, stop_event, results, test_num)

def main():
    def set_model():
        nonlocal model, model_path
        model, model_path = load_model()
        model_path_var.set(f"Модель: {model_path}")
        update_start_button_state()
    
    def set_save_dir():
        nonlocal save_dir
        save_dir = select_save_directory()
        save_dir_path_var.set(f"Збереження: {save_dir}")
        update_start_button_state()

    def start_evaluation():
        nonlocal evaluating_thread, stop_event
        start_button.config(state=DISABLED)
        stop_button.config(state=NORMAL)
        test_num = int(test_num_var.get())
        stop_event = threading.Event()
        evaluating_thread = threading.Thread(target=evaluate_multiple_games, args=(model, model_path, save_dir, 5, stop_event, test_num))
        evaluating_thread.start()

    def stop_evaluation():
        nonlocal stop_event
        stop_event.set()
        evaluating_thread.join()
        start_button.config(state=NORMAL)
        stop_button.config(state=DISABLED)

    def update_start_button_state():
        if model and save_dir:
            start_button.config(state=NORMAL)
        else:
            start_button.config(state=DISABLED)

    root = Tk()
    root.title("Оцінка моделі")
    root.geometry("500x250")

    model = None
    model_path = None
    save_dir = None
    evaluating_thread = None
    stop_event = None

    model_path_var = StringVar()
    save_dir_path_var = StringVar()

    model_button = Button(root, text="Завантажити модель", command=set_model)
    model_button.pack()

    save_dir_button = Button(root, text="Вибрати директорію для збереження", command=set_save_dir)
    save_dir_button.pack()

    model_path_label = Label(root, textvariable=model_path_var)
    model_path_label.pack()

    save_dir_path_label = Label(root, textvariable=save_dir_path_var)
    save_dir_path_label.pack()

    test_num_var = StringVar()
    test_num_entry = Entry(root, textvariable=test_num_var)
    test_num_entry.pack()
    test_num_var.set("1")

    test_num_label = Label(root, text="Введіть номер тесту")
    test_num_label.pack()

    start_button = Button(root, text="Почати оцінку", command=start_evaluation)
    start_button.pack()

    stop_button = Button(root, text="Завершити тест", command=stop_evaluation, state=DISABLED)
    stop_button.pack()
    update_start_button_state()

    root.mainloop()

if __name__ == "__main__":
    main()
