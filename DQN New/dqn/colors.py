class Colors:
    COLORS = dict(black=0, red=1, green=2, yellow=3, blue=4, magenta=5, cyan=6, white=7, default=9, gold=15)
    COLORS_NUMS = inv_map = {v: k for k, v in COLORS.items()}

    BLACK = COLORS['black']
    RED = COLORS['red']
    GREEN = COLORS['green']
    YELLOW = COLORS['yellow']
    BLUE = COLORS['blue']
    MAGENTA = COLORS['magenta']
    CYAN = COLORS['cyan']
    WHITE = COLORS['white']
    GOLD = COLORS['gold']
    DEFAULT = COLORS['default']