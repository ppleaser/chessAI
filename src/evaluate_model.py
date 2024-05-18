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

def evaluate_model(model, model_path, test_num, save_dir, stop_event):
    model_improvements = []
    stockfish_improvements = []

    board = chess.Board()
    first_move = True
    print(f"Тест №{test_num}")

    move_counter = 0
    model_scores = []
    stockfish_scores = []

    while not board.is_game_over() and not stop_event.is_set():
        eval_score_before_stockfish = get_stockfish_evaluation(board, chess.WHITE)
        neural_net_move = get_neural_net_move(model, board)
        board.push(neural_net_move)
      
        eval_score_after_model = get_stockfish_evaluation(board, chess.WHITE)
        board.pop()

        model_improvement = eval_score_after_model - eval_score_before_stockfish
        model_improvements.append(model_improvement)

        stockfish_move = get_stockfish_move(board, first_move)
        first_move = False
        board.push(stockfish_move)

        eval_score_after_stockfish = get_stockfish_evaluation(board, chess.WHITE)

        stockfish_improvement = eval_score_after_stockfish - eval_score_before_stockfish
        stockfish_improvements.append(stockfish_improvement)

        stockfish_move = get_stockfish_move(board, first_move)
        board.push(stockfish_move)

        print(f"Оцінка від Stockfish до: {eval_score_before_stockfish}")
        print(f"Оцінка від Stockfish після ходу моделі: {eval_score_after_model}")
        print(f"Оцінка Stockfish після свого ходу: {eval_score_after_stockfish}")

        move_counter += 1
        model_scores.append(model_improvement)
        stockfish_scores.append(stockfish_improvement)

        mean_model_improvement = np.mean(model_improvements)
        mean_stockfish_improvement = np.mean(stockfish_improvements)
        correlation = np.corrcoef(model_improvements, stockfish_improvements)[0, 1] if len(model_improvements) > 1 else 0

        update_plot(move_counter, model_scores, stockfish_scores, model_path, test_num, mean_model_improvement, mean_stockfish_improvement, correlation)

    print(f"Середнє покращення від ходів моделі: {mean_model_improvement}")
    print(f"Середнє покращення від ходів Stockfish: {mean_stockfish_improvement}")
    print(f"Кореляція між покращеннями моделі та Stockfish: {correlation}")
    print(f"Гра завершена. Результат: {board.result()}")

    save_plot(model_path, test_num, save_dir)

def update_plot(move_counter, model_scores, stockfish_scores, model_path, test_num, mean_model_improvement, mean_stockfish_improvement, correlation):
    plt.clf()
    plt.plot(range(move_counter), model_scores, label='Покращення моделі')
    plt.plot(range(move_counter), stockfish_scores, label='Покращення Stockfish')
    plt.xlabel('Ходи')
    plt.ylabel('Оцінки')
    plt.legend()
    plt.ylim(-10, max(max(model_scores), max(stockfish_scores)) + 1)
    plt.title(f"Тест №{test_num} - Модель: {os.path.basename(model_path)}")
    plt.figtext(0.15, 0.85, f"Середнє покращення моделі: {mean_model_improvement:.2f}")
    plt.figtext(0.15, 0.80, f"Середнє покращення Stockfish: {mean_stockfish_improvement:.2f}")
    plt.figtext(0.15, 0.75, f"Кореляція: {correlation:.2f}")
    plt.draw()
    plt.pause(0.01)

def save_plot(model_path, test_num, save_dir):
    plt.savefig(os.path.join(save_dir, f"game_{test_num}_{os.path.basename(model_path)}.png"))

def load_model():
    file_path = filedialog.askdirectory()
    model = models.load_model(file_path)

    if not model._is_compiled:
        model.compile(optimizer="adam", loss="mean_squared_error")
    
    return model, file_path

def select_save_directory():
    return filedialog.askdirectory()

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
        evaluating_thread = threading.Thread(target=evaluate_model, args=(model, model_path, test_num, save_dir, stop_event))
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