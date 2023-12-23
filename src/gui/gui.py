import pygame
import math
import chess

def update_display(display_queue):

    pygame.init()
    pygame.display.set_caption("Chess Game")
    screen = pygame.display.set_mode((1900, 1030), pygame.RESIZABLE)
    light_square_color = (241, 224, 194)
    dark_square_color = (194, 159, 129)

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

def draw_arrow(screen, color, start, end, width = 8):
    pygame.draw.line(screen, color, start, end, width)

    # Расчет направления стрелки
    direction = [end[0] - start[0], end[1] - start[1]]
    vec_len = math.sqrt(direction[0]**2 + direction[1]**2)
    direction[0] /= vec_len
    direction[1] /= vec_len

    # Расчет точек для треугольника стрелки
    arrow_size = 15
    left_point = [end[0] - direction[0]*arrow_size - direction[1]*arrow_size/2, 
                  end[1] - direction[1]*arrow_size + direction[0]*arrow_size/2]
    right_point = [end[0] - direction[0]*arrow_size + direction[1]*arrow_size/2, 
                   end[1] - direction[1]*arrow_size - direction[0]*arrow_size/2]

    pygame.draw.polygon(screen, color, [end, left_point, right_point])


def draw_text(screen, text_surface, background_rect, text, color, position, font, background_color):
    if text_surface is not None:
        pygame.draw.rect(screen, background_color, background_rect)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    background_rect = text_rect.copy()
    screen.blit(text_surface, text_rect)
    return text_surface, background_rect

def draw_board(board, last_move=None, process_number = 0, neural_net_color = "", eval_score_before = 0.0):
    board_info = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color_prefix = "w" if piece.color == chess.WHITE else "b"
            piece_name = f"{color_prefix}{piece.symbol().upper()}"
            image_path = f"images/{piece_name}.png"
            position = (chess.square_file(square) * 50, (7 - chess.square_rank(square)) * 50)
            board_info.append((image_path, position))

    if last_move is not None:
        start_position = (chess.square_file(last_move.from_square) * 50 + 25, (7 - chess.square_rank(last_move.from_square)) * 50 + 25)
        end_position = (chess.square_file(last_move.to_square) * 50 + 25, (7 - chess.square_rank(last_move.to_square)) * 50 + 25)
        board_info.append(("highlight", start_position, end_position))

    return board_info, process_number, neural_net_color, eval_score_before, board, last_move