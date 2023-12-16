import pygame
import math
import chess

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