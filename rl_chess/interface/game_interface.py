import pygame
import pathlib
from rl_chess import base_path

pygame.init()
pygame.font.init()

my_font = pygame.font.SysFont('Courier New', 32, bold=True)

character_images = {
    char: my_font.render(char, False, (0, 0, 0))
    for char in "abcdefgh12345678"
}

font_padding = 64
screen = pygame.display.set_mode([512 + font_padding, 512 + font_padding])

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

piece_images: dict[str, pygame.surface.Surface] = {
    piece: pygame.image.load(str(base_path / path))
    for piece, path in symbol_to_image_path.items()
}
