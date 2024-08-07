import functools
import logging
import random
import time
from typing import Literal

import chess
import pygame as pg
import pygame_widgets.button

from rl_chess import base_path
from rl_chess.inference.inference import ChessAgent

logger = logging.getLogger(__name__)


class GameInterface:

    FONT_PADDING = 64
    SQUARE_SIZE = 64
    LIGHT_SQUARE_COLOR = (255, 255, 255)
    DARK_SQUARE_COLOR = (128, 128, 128)
    SELECTED_SQUARE_TINT = (0, 255, 0)
    LEGAL_MOVE_TINT = (64, 192, 64)

    def __init__(
        self,
        font: pg.font.Font | None = None,
    ) -> None:
        pass

        self.font = font or pg.font.SysFont("Courier New", 32, bold=True)
        self.screen = pg.display.set_mode(
            [
                self.SQUARE_SIZE * 8 + self.FONT_PADDING,
                self.SQUARE_SIZE * 8 + self.FONT_PADDING,
            ]
        )
        self.piece_images = self._load_images()

    def _load_images(self) -> dict[str, pg.surface.Surface]:
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
            piece: pg.image.load(str(base_path / path))
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

    def get_victory_banner(self, outcome: chess.Outcome) -> pg.surface.Surface:
        banner = pg.Surface((512, 512))
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
            pg.draw.rect(
                self.screen,
                color,
                pg.Rect(
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * col,
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * (7 - row),
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                ),
            )
        # draw legend
        # check if we need to flip the board
        for idx, rank in enumerate(
            chess.RANK_NAMES[:: -1 if perspective == chess.WHITE else 1]
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
        pg.draw.rect(
            self.screen,
            self.mix_color(highlight_tint, orig_square_color),
            pg.Rect(
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

    def select_color(self) -> chess.Color:
        selected_color: chess.Color | str | None = None

        def on_click(action: Literal["white", "black", "random"]) -> None:
            nonlocal selected_color
            match action:
                case "white":
                    selected_color = chess.WHITE
                case "black":
                    selected_color = chess.BLACK
                case "random":
                    selected_color = random.choice([chess.WHITE, chess.BLACK])

        white_button = pygame_widgets.button.Button(
            self.screen,
            x=150,
            y=50,
            width=250,
            height=125,
            text="white",
            fontSize=50,
            margin=20,
            inactiveColour=(255, 255, 255),
            pressedColour=(64, 64, 64),
            radius=5,
            onClick=functools.partial(on_click, "white"),
        )

        black_button = pygame_widgets.button.Button(
            self.screen,
            x=150,
            y=200,
            width=250,
            height=125,
            text="black",
            fontSize=50,
            margin=20,
            inactiveColour=(255, 255, 255),
            pressedColour=(64, 64, 64),
            radius=5,
            onClick=functools.partial(on_click, "black"),
        )

        random_button = pygame_widgets.button.Button(
            self.screen,
            x=150,
            y=350,
            width=250,
            height=125,
            text="random",
            fontSize=50,
            margin=20,
            inactiveColour=(255, 255, 255),
            pressedColour=(64, 64, 64),
            radius=5,
            onClick=functools.partial(on_click, "random"),
        )

        while selected_color is None:
            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    selected_color = chess.WHITE
            pygame_widgets.update(events)
            white_button.draw()
            black_button.draw()
            random_button.draw()
            pg.display.update()
        return selected_color

    def display_move_scores(
        self, board: chess.Board, move_scores: dict[chess.Move, float]
    ) -> None:
        """
        Highlight the top-rated moves on the board with a gradient of colors outlining the squares.
        Blue indicates a low score, red indicates a high score.
        """
        max_score = max(move_scores.values())
        min_score = min(move_scores.values())
        for move, score in move_scores.items():
            square = move.to_square
            row, col = chess.square_rank(square), chess.square_file(square)
            if board.turn == chess.BLACK:
                row = 7 - row
                col = 7 - col
            orig_square_color = (
                self.LIGHT_SQUARE_COLOR
                if (row + col) % 2 == 0
                else self.DARK_SQUARE_COLOR
            )
            scaled_score = (score - min_score) / (max_score - min_score + 1e-6)
            tint = self.mix_color((0, 0, 255), (255, 0, 0), scaled_score)
            pg.draw.rect(
                self.screen,
                self.mix_color(tint, orig_square_color),
                pg.Rect(
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * col,
                    self.FONT_PADDING / 2 + self.SQUARE_SIZE * (7 - row),
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                ),
                4,
            )

    def update_display(
        self,
        board: chess.Board,
        selected_square: chess.Square | None = None,
        move_scores: dict[chess.Move, float] | None = None,
    ) -> None:
        self.draw_blank_board(board.turn)
        if selected_square is not None:
            self.draw_selected(board, selected_square)
            self.draw_legal_moves(board, selected_square)
        if move_scores is not None:
            self.display_move_scores(board, move_scores)
        self.draw_pieces(board)
        if board.is_game_over():
            outcome = board.outcome()
            banner = self.get_victory_banner(outcome)
            self.screen.blit(banner, (self.FONT_PADDING / 2, self.FONT_PADDING / 2))
        pg.display.flip()


class GameRunner:
    def __init__(
        self,
        board: chess.Board | None = None,
        chess_agent: ChessAgent | None = None,
        game_interface: GameInterface | None = None,
    ) -> None:
        pg.init()
        pg.font.init()

        self.board = board or chess.Board()
        self.chess_agent = chess_agent or ChessAgent()
        self.game_interface = game_interface or GameInterface()
        self.move_scores: dict[chess.Move, float] | None = None
        self.selected_square = None
        self.running = True

    def create_move(
        self, board: chess.Board, start_square: chess.Square, end_square: chess.Square
    ) -> chess.Move:
        # Check for promotion
        if board.piece_at(start_square) == chess.Piece(chess.PAWN, board.turn):
            if chess.square_rank(end_square) in (0, 7):
                return chess.Move(start_square, end_square, promotion=chess.QUEEN)
        return chess.Move(start_square, end_square)

    def handle_mouseup(self, xpos: int, ypos: int) -> None:
        logger.debug(f"Mouse up at ({xpos}, {ypos})")
        square = self.game_interface.get_square(xpos, ypos, self.board.turn)
        if square is None:
            logger.debug("Mouse up off the board")
        elif self.selected_square is None:
            # select a new piece
            piece = self.board.piece_at(square)
            if piece is not None and self.board.turn == piece.color:
                logger.info(f"Selected piece: {piece} at {square}")
                self.move_scores = None
                self.selected_square = square
        else:
            # move the selected piece (if the move is legal)
            move = self.create_move(self.board, self.selected_square, square)
            if move in self.board.legal_moves:
                logger.info(f"Moving piece from {self.selected_square} to {square}")
                self.board.push(move)
            else:
                logger.info(f"Selected invalid move: {move}")
            self.selected_square = None
            self.move_scores = None

    def ai_move(self, board: chess.Board) -> None:
        move = self.chess_agent.select_top_rated_move(board)
        board.push(move)
        time.sleep(1)

    def run(self) -> None:
        player_color = self.game_interface.select_color()
        while self.running:
            self.game_interface.update_display(
                board=self.board,
                selected_square=self.selected_square,
                move_scores=self.move_scores,
            )
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logger.info("Quitting game")
                    self.running = False
                elif event.type == pg.MOUSEBUTTONUP and self.board.turn == player_color:
                    self.handle_mouseup(*pg.mouse.get_pos())
            if self.board.turn != player_color and not self.board.is_game_over():
                self.ai_move(self.board)
            if (
                self.selected_square is not None
                and self.board.turn == player_color
                and self.move_scores is None
            ):
                # Calculate move scores if a piece is selected
                logger.info(f"Calculating move scores for {self.selected_square}")
                self.move_scores = self.chess_agent.rate_moves_from_position(
                    self.board, self.selected_square
                )
        pg.quit()
