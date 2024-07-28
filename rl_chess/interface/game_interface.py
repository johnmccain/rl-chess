import pathlib

import chess
import pygame

from rl_chess import base_path


class GameInterface:

    FONT_PADDING = 64
    SQUARE_SIZE = 64
    LIGHT_SQUARE_COLOR = (255, 255, 255)
    DARK_SQUARE_COLOR = (128, 128, 128)
    SELECTED_SQUARE_TINT = (0, 255, 0)
    LEGAL_MOVE_TINT = (64, 192, 64)

    def __init__(
        self,
        font: pygame.font.Font | None = None,
    ) -> None:
        pass

        self.font = font or pygame.font.SysFont("Courier New", 32, bold=True)
        self.screen = pygame.display.set_mode(
            [
                self.SQUARE_SIZE * 8 + self.FONT_PADDING,
                self.SQUARE_SIZE * 8 + self.FONT_PADDING,
            ]
        )
        self.piece_images = self._load_images()

    def _load_images(self) -> dict[str, pygame.surface.Surface]:
        # load images
        symbol_to_image_path = {
            "b": "img/black_bishop.png",
            "k": "img/black_king.png",
            "n": "img/black_knight.png",
            "p": "img/black_pawn.png",
            "q": "img/black_queen.png",
            "r": "img/black_rook.png",
            "B": "img/white_bishop.png",
            "K": "img/white_king.png",
            "N": "img/white_knight.png",
            "P": "img/white_pawn.png",
            "Q": "img/white_queen.png",
            "R": "img/white_rook.png",
        }

        piece_images = {
            piece: pygame.image.load(str(base_path / path))
            for piece, path in symbol_to_image_path.items()
        }
        return piece_images

    @staticmethod
    def mix_color(
        color1: tuple[int, int, int], color2: tuple[int, int, int], alpha: float = 0.5
    ) -> tuple[int, int, int]:
        """
        :param color1: tuple of 3 ints representing an RGB color
        :param color2: tuple of 3 ints representing an RGB color
        :param alpha: float between 0 and 1 representing the weight of color1
        """
        return tuple(int(color1[i] * alpha + color2[i] * (1 - alpha)) for i in range(3))

    def get_victory_banner(self, outcome: chess.Outcome) -> pygame.surface.Surface:
        banner = pygame.Surface((512, 512))
        banner.set_alpha(192)
        if outcome.winner == chess.WHITE:
            banner.fill((255, 255, 255))
            banner.blit(self.font.render("White wins!", False, (0, 0, 0)), (128, 128))
        elif outcome.winner == chess.BLACK:
            banner.fill((0, 0, 0))
            banner.blit(
                self.font.render("Black wins!", False, (255, 255, 255)), (128, 128)
            )
        else:
            banner.fill((192, 192, 192))
            banner.blit(self.font.render("Draw!", False, (0, 0, 0)), (128, 128))
        return banner

    def get_square(self, x: int, y: int, turn: chess.Color) -> chess.Square | None:
        """
        Get the chess square corresponding to the pixel coordinates (x, y) (or None if the pixel is not on the board)
        :param x: x coordinate of the pixel
        :param y: y coordinate of the pixel
        :param turn: the color of the player whose perspective we are looking from
        """
        row = int(7 - (y - self.FONT_PADDING / 2) // self.SQUARE_SIZE)
        col = int((x - self.FONT_PADDING / 2) // self.SQUARE_SIZE)
        if row < 0 or row >= 8 or col < 0 or col >= 8:
            return None
        if turn == chess.BLACK:
            row = 7 - row
            col = 7 - col
        return chess.square(col, row)

    def draw_blank_board(self, perspective: chess.Color) -> None:
        # blank the screen
        self.screen.fill((255, 255, 255))

        # draw squares
        for square in chess.SQUARES[:: 1 if perspective == chess.WHITE else -1]:
            row, col = chess.square_rank(square), chess.square_file(square)
            if (row + col) % 2 == 0:
                color = self.LIGHT_SQUARE_COLOR
            else:
                color = self.DARK_SQUARE_COLOR
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * col,
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * (7 - row),
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                ),
            )
        # draw legend
        # check if we need to flip the board
        for idx, rank in enumerate(
            chess.RANK_NAMES[:: 1 if perspective == chess.WHITE else -1]
        ):
            text = self.font.render(rank, False, (0, 0, 0))
            self.screen.blit(
                text,
                (
                    self.FONT_PADDING / 2 - 32,
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * idx,
                ),
            )
            self.screen.blit(
                text,
                (
                    8 * self.SQUARE_SIZE + self.FONT_PADDING / 2,
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * idx,
                ),
            )
        for idx, file in enumerate(chess.FILE_NAMES):
            text = self.font.render(file, False, (0, 0, 0))
            self.screen.blit(text, (self.FONT_PADDING / 2 + self.SQUARE_SIZE * idx, 0))
            self.screen.blit(
                text,
                (
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * idx,
                    8 * self.SQUARE_SIZE + self.FONT_PADDING / 2,
                ),
            )

    def draw_highlight(
        self,
        square: chess.Square,
        highlight_tint: tuple[int, int, int],
        turn: chess.Color,
    ) -> None:
        row, col = chess.square_rank(square), chess.square_file(square)
        if turn == chess.BLACK:
            row = 7 - row
            col = 7 - col
        orig_square_color = (
            self.LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else self.DARK_SQUARE_COLOR
        )
        pygame.draw.rect(
            self.screen,
            self.mix_color(highlight_tint, orig_square_color),
            pygame.Rect(
                self.FONT_PADDING / 2 + self.SQUARE_SIZE * col,
                self.FONT_PADDING / 2 + self.SQUARE_SIZE * (7 - row),
                self.SQUARE_SIZE,
                self.SQUARE_SIZE,
            ),
            4,
        )

    def draw_selected(self, board: chess.Board, square: chess.Square) -> None:
        self.draw_highlight(square, self.SELECTED_SQUARE_TINT, board.turn)

    def draw_legal_moves(self, board: chess.Board, square: chess.Square) -> None:
        for move in board.legal_moves:
            if move.from_square == square:
                self.draw_highlight(move.to_square, self.LEGAL_MOVE_TINT, board.turn)

    def draw_pieces(self, board: chess.Board) -> None:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                row, col = chess.square_rank(square), chess.square_file(square)
                if board.turn == chess.BLACK:
                    row = 7 - row
                    col = 7 - col
                image = self.piece_images[piece.symbol()]
                self.screen.blit(
                    image,
                    (
                        self.FONT_PADDING / 2 + self.SQUARE_SIZE * col,
                        self.FONT_PADDING / 2 + self.SQUARE_SIZE * (7 - row),
                    ),
                )

    def update_display(
        self,
        board: chess.Board,
        selected_square: chess.Square | None = None,
    ) -> None:
        self.draw_blank_board(board.turn)
        if selected_square is not None:
            self.draw_selected(board, selected_square)
            self.draw_legal_moves(board, selected_square)
        self.draw_pieces(board)
        if board.is_game_over():
            outcome = board.outcome()
            banner = self.get_victory_banner(outcome)
            self.screen.blit(banner, (self.FONT_PADDING / 2, self.FONT_PADDING / 2))
        pygame.display.flip()


class GameRunner:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()

        self.board = chess.Board()
        self.game_interface = GameInterface()
        self.selected_square = None
        self.running = True

    def handle_mouseup(self, xpos: int, ypos: int) -> None:
        xpos, ypos = pygame.mouse.get_pos()
        square = self.game_interface.get_square(xpos, ypos, self.board.turn)
        if self.selected_square is None:
            # select a new piece
            piece = self.board.piece_at(square)
            if piece is not None and self.board.turn == piece.color:
                self.selected_square = square
        else:
            # move the selected piece (if the move is legal)
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
            self.selected_square = None

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouseup(*pygame.mouse.get_pos())
            self.game_interface.update_display(
                board=self.board, selected_square=self.selected_square
            )

        pygame.quit()
