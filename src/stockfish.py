import pygame
import chess
import chess.svg
import chess.engine
import random
import numpy as np
import os
import threading
import signal
import sys

from tensorflow.keras import layers, models, Model, Input
from multiprocessing import Process, Manager, freeze_support
from pygame.locals import *
from gui.gui import draw_arrow, draw_text, draw_board


last_model_number = 0
num_training_cycles = 200


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer class.

        Parameters:
        - capacity (int): The maximum capacity of the replay buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.

        Parameters:
        - state: The current state of the environment.
        - action: The action taken in the current state.
        - reward: The reward received for taking the action.
        - next_state: The next state of the environment.
        - done: A flag indicating if the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.

        Parameters:
        - batch_size (int): The size of the batch to sample.

        Returns:
        - state: The states in the batch.
        - action: The actions in the batch.
        - reward: The rewards in the batch.
        - next_state: The next states in the batch.
        - dones: The done flags in the batch.
        """
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, dones = map(np.array, zip(*batch))
        return state, action, reward, next_state, dones

    def __len__(self):
        """
        Get the current size of the replay buffer.

        Returns:
        - int: The size of the replay buffer.
        """
        return len(self.buffer)


def train_neural_net(model, replay_buffer, batch_size):
    """
    Trains a neural network model using a replay buffer.

    Args:
        model (object): The neural network model to be trained.
        replay_buffer (object): The replay buffer containing training data.
        batch_size (int): The size of each training batch.

    Returns:
        object: The trained neural network model.
    """
    replay_buffer_for_training = ReplayBuffer(10000)
    replay_buffer_for_training.buffer = list(replay_buffer)

    states, actions, rewards, next_states, dones = replay_buffer_for_training.sample(
        batch_size
    )

    target_values = rewards + (1 - dones) * model.predict([next_states, actions])[:, 0]
    target = model.predict([states, actions])
    target[:, 0] = target_values

    model.train_on_batch([states, actions], target)
    print("Модель успешно прошла тренировку")
    return model


def play_game(model, neural_net_color, display_queue, i, replay_buffer):
    """
    Play a game of chess between a neural network model and Stockfish engine.

    Args:
        model (NeuralNetwork): The neural network model.
        neural_net_color (chess.Color): The color of the neural network player.
        display_queue (Queue): The queue for displaying the chess board.
        i (int): The index of the game.
        replay_buffer (list): The replay buffer for training the neural network.

    Returns:
        NeuralNetwork: The updated neural network model.
    """
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
            changes_buffer = draw_board(
                board, last_move, i, neural_net_color, eval_score_before
            )
            display_queue.put(changes_buffer)
        else:
            stockfish_move = get_stockfish_move(board, first_move)
            first_move = False
            board.push(stockfish_move)
            last_move = stockfish_move
            changes_buffer = draw_board(
                board, last_move, i, neural_net_color, eval_score_before
            )
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

    print("Игра была закончена")

    model = train_neural_net(model, replay_buffer, 64)

    with open(
        f"models/reinforcement_learning_stockfish_model/game_results.txt",
        "a",
        encoding="utf-8",
    ) as file:
        file.write(f"Итог партии: {board.result()}, ")
        file.write(
            f"Игра за {'белых' if neural_net_color == chess.WHITE else 'черных'}, "
        )
        file.write(
            f"Процент правильных ходов от нейросети: {correct_moves / total_moves * 100:.2f}%, "
        )
        file.write(f"История ходов нейросети: {', '.join(neural_net_moves)}, ")
        file.write(f"История ходов Stockfish: {', '.join(stockfish_moves)}, ")
        file.write(
            f"Награды и штрафы после каждого хода: {', '.join(map(str, rewards))}\n"
        )

    return model


def get_neural_net_move(model, board, exploration_factor=0.1):
    """
    Returns the best move predicted by a neural network model for the given chess board.

    Parameters:
        model (object): The neural network model used for move prediction.
        board (object): The chess board object representing the current game state.
        exploration_factor (float): The probability of making a random move instead of the predicted move.

    Returns:
        object: The best move predicted by the neural network model.
    """
    legal_moves = list(board.legal_moves)

    if (
        np.random.rand() < exploration_factor
        or board.empty
        or board.fullmove_number == 1
    ):
        return random.choice(legal_moves)

    board_copy = board.copy()  # Создаем копию доски

    move_predictions = []

    for move in legal_moves:
        board_copy.push(move)
        board_tensor = board_to_tensor(board_copy)
        move_tensor = move_to_tensor(move)
        input_tensor = np.concatenate([board_tensor, move_tensor], axis=-1)
        prediction = model.predict(input_tensor)[0][0]
        move_predictions.append((move, prediction))
        board_copy.pop()  # Отменяем ход

    sorted_moves = sorted(move_predictions, key=lambda x: x[1], reverse=True)
    return sorted_moves[0][0]


def get_stockfish_move(board, first_move=False):
    """
    Get the best move for the given chess board using Stockfish engine.

    Parameters:
    - board (chess.Board): The chess board to get the move for.
    - first_move (bool): Flag indicating if it's the first move. If True, a random legal move will be returned.

    Returns:
    - move (chess.Move): The best move suggested by Stockfish engine.
    """
    if first_move:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

    stockfish_path = "stockfish/stockfish.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.play(board, chess.engine.Limit(depth=20, time=0.5))
    move = result.move
    engine.quit()
    return move


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


def average_models(models):
    """
    Averages the weights of a list of models and returns a new model with the averaged weights.

    Parameters:
    models (list): A list of models to be averaged.

    Returns:
    averaged_model: A new model with the averaged weights.
    """
    averaged_model = create_model()

    weights = [model.get_weights() for model in models]
    averaged_weights = [sum(w) / len(w) for w in zip(*weights)]

    averaged_model.set_weights(averaged_weights)

    return averaged_model


def create_model():
    """
    Creates a convolutional neural network model for the chess AI.

    Returns:
        model (tensorflow.keras.Model): The compiled model.
    """
    state_input = Input(shape=(8, 8, 13), name="state")
    action_input = Input(shape=(8, 8, 13), name="action")

    x = layers.Concatenate()([state_input, action_input])

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    output = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def run_game_for_color(color, result_queue, display_queue, i, replay_buffer):
    """
    Runs a game for a specified color using a pre-trained model.

    Args:
        color (str): The color of the player ('white' or 'black').
        result_queue (Queue): A queue to store the game result.
        display_queue (Queue): A queue to display the game on the UI.
        i (int): The index of the game.
        replay_buffer (ReplayBuffer): The replay buffer to store the game data.

    Returns:
        None
    """
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

            print(f"Загружена модель: {last_model_number}")
        else:
            raise IOError("В директории не найдены модели.")

        result = play_game(model, color, display_queue, i, replay_buffer)
        result_queue.put(result)
        return
    except (OSError, IOError) as e:
        print(f"Создание новой модели для цикла.")
        model = create_model()
    except KeyboardInterrupt:
        print(f"Process {os.getpid()} received KeyboardInterrupt. Terminating...")
        result_queue.put(None)
        sys.exit(0)


def update_display(display_queue):
    """
    Update the display with the current state of the chess game.

    Args:
        display_queue (Queue): A queue containing the information needed to update the display.

    Returns:
        None
    """
    pygame.init()
    pygame.display.set_caption("Chess Game")
    screen = pygame.display.set_mode((1900, 1030), pygame.RESIZABLE)
    light_square_color = (240, 240, 240)
    dark_square_color = (100, 100, 100)

    piece_images = {}
    text_surface_top = [[None] * 4 for _ in range(2)]  # Два ряда для каждой стороны
    text_surface_bottom = [[None] * 4 for _ in range(2)]
    background_rect_top = [[None] * 4 for _ in range(2)]
    background_rect_bottom = [[None] * 4 for _ in range(2)]
    clock = pygame.time.Clock()
    while True:
        pygame.event.pump()
        if not display_queue.empty():
            (
                board_info,
                process_number,
                neural_net_color,
                eval_score_before,
                board,
                last_move,
            ) = display_queue.get()

            for piece_info in board_info:
                image_path, position = piece_info[:2]
                if image_path not in piece_images and image_path != "highlight":
                    piece_images[image_path] = pygame.transform.scale(
                        pygame.image.load(image_path), (50, 50)
                    )

            x_offset = ((screen.get_width() - (4 * 400 + 3 * 50)) / 2) + (
                process_number % 4 * (400 + 50)
            )
            y_offset = 0 if process_number < 4 else screen.get_height() / 2

            for row in range(8):
                for col in range(8):
                    color = (
                        light_square_color
                        if (row + col) % 2 == 0
                        else dark_square_color
                    )
                    pygame.draw.rect(
                        screen,
                        color,
                        (col * 50 + x_offset, row * 50 + y_offset + 50, 50, 50),
                    )

            for piece_info in board_info:
                image_path, position = piece_info[:2]
                if image_path == "highlight":
                    start_position, end_position = position, piece_info[2]
                    if (
                        board.piece_type_at(last_move.to_square) == chess.KNIGHT
                    ):  # Если ход совершает конь
                        if abs(start_position[0] - end_position[0]) > abs(
                            start_position[1] - end_position[1]
                        ):
                            intermediate_position = (end_position[0], start_position[1])
                        else:
                            intermediate_position = (start_position[0], end_position[1])
                        pygame.draw.line(
                            screen,
                            (0, 255, 0),
                            (
                                start_position[0] + x_offset,
                                start_position[1] + y_offset + 50,
                            ),
                            (
                                intermediate_position[0] + x_offset,
                                intermediate_position[1] + y_offset + 50,
                            ),
                            3,
                        )
                        draw_arrow(
                            screen,
                            (0, 255, 0),
                            (
                                intermediate_position[0] + x_offset,
                                intermediate_position[1] + y_offset + 50,
                            ),
                            (
                                end_position[0] + x_offset,
                                end_position[1] + y_offset + 50,
                            ),
                            3,
                        )
                    else:
                        draw_arrow(
                            screen,
                            (0, 255, 0),
                            (
                                start_position[0] + x_offset,
                                start_position[1] + y_offset + 50,
                            ),
                            (
                                end_position[0] + x_offset,
                                end_position[1] + y_offset + 50,
                            ),
                            3,
                        )
                else:
                    screen.blit(
                        piece_images[image_path],
                        (position[0] + x_offset, position[1] + y_offset + 50),
                    )

            # Добавляем текст над и под каждой доской
            font = pygame.font.Font(None, 32)
            text_color = (255, 255, 255)
            background_color = (0, 0, 0)  # Цвет фона

            text_top = (
                f"Stockfish ({-eval_score_before:.2f})"
                if neural_net_color == chess.WHITE
                else f"Neural Net ({eval_score_before:.2f})"
            )
            text_bottom = (
                f"Stockfish ({-eval_score_before:.2f})"
                if neural_net_color == chess.BLACK
                else f"Neural Net ({eval_score_before:.2f})"
            )

            (
                text_surface_top[neural_net_color][process_number % 4],
                background_rect_top[neural_net_color][process_number % 4],
            ) = draw_text(
                screen,
                text_surface_top[neural_net_color][process_number % 4],
                background_rect_top[neural_net_color][process_number % 4],
                text_top,
                text_color,
                (x_offset + 200, y_offset + 25),
                font,
                background_color,
            )

            (
                text_surface_bottom[neural_net_color][process_number % 4],
                background_rect_bottom[neural_net_color][process_number % 4],
            ) = draw_text(
                screen,
                text_surface_bottom[neural_net_color][process_number % 4],
                background_rect_bottom[neural_net_color][process_number % 4],
                text_bottom,
                text_color,
                (x_offset + 200, y_offset + 8 * 50 + 75),
                font,
                background_color,
            )

            pygame.display.flip()
            clock.tick(120)


if __name__ == "__main__":
    try:
        freeze_support()

        with Manager() as manager:
            replay_buffer = manager.list()
            result_queue = manager.Queue()
            display_queue = manager.Queue()
            processes = []

            display_thread = threading.Thread(
                target=update_display, args=(display_queue,)
            )
            display_thread.daemon = True
            display_thread.start()

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
                    results = [result_queue.get() for _ in processes]
                    my_models = [result for result in results if result is not None]

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
                        averaged_model = train_neural_net(
                            averaged_model, replay_buffer, 64
                        )
                        averaged_model.save(new_model_path)
                        print(f"Сохранена модель: {new_model_number}")

                processes = []
                my_models = []
                results = []
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        pygame.quit()
        exit()
