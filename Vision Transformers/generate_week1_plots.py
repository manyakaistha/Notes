"""
generate_week1_plots.py — Deterministic figure generation for Vision Transformers Week 1.
Generates all publication-grade figures for Week 1 (Day 1 & Day 2):
Day 1:
1. image_channels_pixels.png
2. convolution_feature_extraction.png
3. padding_and_pooling.png
4. cnn_feature_vector_pipeline.png
5. receptive_field_vs_vanishing_gradient.png
6. cnn_limitations_and_long_range_dependency.png
7. vision_transformer_architecture.png
Day 2:
8. tokenizer_paradigms_bpe.png
9. token_and_position_embeddings.png
10. transformer_block_architecture.png
11. layer_norm_and_dropout.png
12. ffn_expansion_and_skip_connections.png
13. output_projection_and_next_token.png
"""

import os
import struct
import zlib
from pathlib import Path

OUT = Path("month 1/week 1/assets")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lightweight Pure-Python Canvas & PNG Renderer
# ---------------------------------------------------------------------------

class Canvas:
    def __init__(self, width, height, bg=(255, 255, 255)):
        self.width = width
        self.height = height
        self.buffer = bytearray(width * height * 3)
        for i in range(width * height):
            self.buffer[i * 3] = bg[0]
            self.buffer[i * 3 + 1] = bg[1]
            self.buffer[i * 3 + 2] = bg[2]

    def set_pixel(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.buffer[idx] = color[0]
            self.buffer[idx + 1] = color[1]
            self.buffer[idx + 2] = color[2]

    def fill_rect(self, x0, y0, w, h, color):
        x_start = max(0, int(x0))
        x_end = min(self.width, int(x0 + w))
        y_start = max(0, int(y0))
        y_end = min(self.height, int(y0 + h))
        for y in range(y_start, y_end):
            row_idx = (y * self.width + x_start) * 3
            self.buffer[row_idx : row_idx + (x_end - x_start) * 3] = bytearray(color * (x_end - x_start))

    def stroke_rect(self, x0, y0, w, h, color, thickness=1):
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)
        self.fill_rect(x0, y0, w, thickness, color)
        self.fill_rect(x0, y0 + h - thickness, w, thickness, color)
        self.fill_rect(x0, y0, thickness, h, color)
        self.fill_rect(x0 + w - thickness, y0, thickness, h, color)

    def draw_line(self, x0, y0, x1, y1, color, thickness=1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            for tx in range(-thickness // 2, thickness // 2 + 1):
                for ty in range(-thickness // 2, thickness // 2 + 1):
                    self.set_pixel(x + tx, y + ty, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def draw_grid(self, x0, y0, rows, cols, cell_size, grid_color=(200, 200, 200), fill_colors=None):
        for r in range(rows):
            for c in range(cols):
                cx = x0 + c * cell_size
                cy = y0 + r * cell_size
                if fill_colors and (r, c) in fill_colors:
                    self.fill_rect(cx, cy, cell_size, cell_size, fill_colors[(r, c)])
                self.stroke_rect(cx, cy, cell_size, cell_size, grid_color, thickness=1)

    def save_png(self, filepath):
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)  # filter type 0 (None)
            raw.extend(self.buffer[y * self.width * 3 : (y + 1) * self.width * 3])

        def chunk(tag, data):
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack('>IIBBBBB', self.width, self.height, 8, 2, 0, 0, 0)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(bytes(raw), level=6)) + chunk(b'IEND', b'')
        with open(filepath, 'wb') as f:
            f.write(png)

# ---------------------------------------------------------------------------
# Day 1 Figures (1 to 7)
# ---------------------------------------------------------------------------

def generate_fig1():
    w, h = 900, 480
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 20, 820, 50, (230, 242, 255))
    c.stroke_rect(40, 20, 820, 50, (51, 122, 183), thickness=2)
    c.draw_grid(80, 160, 6, 6, 22, grid_color=(180, 50, 50),
                fill_colors={(r, col): (255, 180 + (r*col*3)%70, 180 + (r*col*3)%70) for r in range(6) for col in range(6)})
    c.draw_grid(130, 130, 6, 6, 22, grid_color=(50, 160, 50),
                fill_colors={(r, col): (180 + (r*col*3)%70, 255, 180 + (r*col*3)%70) for r in range(6) for col in range(6)})
    c.draw_grid(180, 100, 6, 6, 22, grid_color=(50, 80, 200),
                fill_colors={(r, col): (180 + (r*col*3)%70, 190 + (r*col*3)%60, 255) for r in range(6) for col in range(6)})
    c.draw_line(340, 170, 430, 170, (80, 80, 80), thickness=3)
    c.draw_line(420, 160, 430, 170, (80, 80, 80), thickness=3)
    c.draw_line(420, 180, 430, 170, (80, 80, 80), thickness=3)
    c.draw_grid(460, 100, 10, 10, 24, grid_color=(160, 160, 160),
                fill_colors={(3, 4): (255, 80, 80), (3, 5): (255, 100, 80), (4, 4): (255, 60, 60), (4, 5): (255, 90, 70)})
    c.stroke_rect(460 + 4*24, 100 + 3*24, 2*24, 2*24, (20, 20, 20), thickness=3)
    c.draw_line(460 + 6*24, 100 + 4*24, 740, 160, (20, 20, 20), thickness=2)
    c.draw_line(460 + 6*24, 100 + 5*24, 740, 260, (20, 20, 20), thickness=2)
    c.fill_rect(740, 160, 100, 100, (255, 75, 75))
    c.stroke_rect(740, 160, 100, 100, (20, 20, 20), thickness=3)
    c.fill_rect(40, 390, 820, 70, (255, 255, 255))
    c.stroke_rect(40, 390, 820, 70, (200, 200, 200), thickness=1)
    c.save_png(OUT / "image_channels_pixels.png")

def generate_fig2():
    w, h = 920, 520
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 840, 45, (230, 245, 230))
    c.stroke_rect(40, 15, 840, 45, (40, 160, 40), thickness=2)
    c.draw_grid(60, 90, 7, 7, 24, grid_color=(180, 180, 180),
                fill_colors={(1, col): (30, 30, 30) for col in range(1, 6)} |
                            {(r, 3): (30, 30, 30) for r in range(1, 6)})
    c.draw_line(240, 140, 310, 110, (100, 100, 100), thickness=2)
    c.draw_grid(320, 80, 3, 3, 26, grid_color=(50, 50, 50),
                fill_colors={(0, col): (255, 200, 200) for col in range(3)} |
                            {(1, col): (240, 240, 240) for col in range(3)} |
                            {(2, col): (200, 255, 200) for col in range(3)})
    c.draw_line(410, 120, 460, 120, (100, 100, 100), thickness=2)
    c.draw_grid(470, 80, 5, 5, 22, grid_color=(180, 180, 180),
                fill_colors={(0, col): (20, 20, 20) for col in range(5)})
    c.draw_line(240, 200, 310, 240, (100, 100, 100), thickness=2)
    c.draw_grid(320, 210, 3, 3, 26, grid_color=(50, 50, 50),
                fill_colors={(r, 0): (255, 200, 200) for r in range(3)} |
                            {(r, 1): (240, 240, 240) for r in range(3)} |
                            {(r, 2): (200, 255, 200) for r in range(3)})
    c.draw_line(410, 250, 460, 250, (100, 100, 100), thickness=2)
    c.draw_grid(470, 210, 5, 5, 22, grid_color=(180, 180, 180),
                fill_colors={(r, 2): (20, 20, 20) for r in range(5)})
    c.fill_rect(60, 340, 800, 150, (255, 255, 255))
    c.stroke_rect(60, 340, 800, 150, (200, 200, 200), thickness=2)
    c.draw_grid(80, 360, 3, 3, 28, grid_color=(100, 100, 255),
                fill_colors={(r, c): (220, 230, 255) for r in range(3) for c in range(3)})
    c.fill_rect(190, 400, 12, 12, (50, 50, 50))
    c.draw_grid(225, 360, 3, 3, 28, grid_color=(255, 100, 100),
                fill_colors={(r, c): (255, 225, 225) for r in range(3) for c in range(3)})
    c.draw_line(330, 400, 370, 400, (50, 50, 50), thickness=3)
    c.fill_rect(390, 375, 50, 50, (220, 255, 220))
    c.stroke_rect(390, 375, 50, 50, (40, 160, 40), thickness=2)
    c.draw_grid(500, 360, 4, 4, 25, grid_color=(180, 180, 180),
                fill_colors={(0, 0): (255, 200, 100), (0, 1): (255, 200, 100), (1, 0): (255, 200, 100), (1, 1): (255, 200, 100)})
    c.draw_line(620, 400, 650, 400, (80, 80, 80), thickness=2)
    c.draw_grid(670, 360, 4, 4, 25, grid_color=(180, 180, 180),
                fill_colors={(0, 1): (255, 200, 100), (0, 2): (255, 200, 100), (1, 1): (255, 200, 100), (1, 2): (255, 200, 100)})
    c.save_png(OUT / "convolution_feature_extraction.png")

def generate_fig3():
    w, h = 940, 540
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 860, 45, (255, 245, 230))
    c.stroke_rect(40, 15, 860, 45, (230, 140, 30), thickness=2)
    c.draw_grid(60, 80, 8, 8, 20, grid_color=(150, 150, 150),
                fill_colors={(r, col): (255, 255, 255) if (r==0 or r==7 or col==0 or col==7) else (210, 230, 255)
                             for r in range(8) for col in range(8)})
    c.stroke_rect(60 + 20, 80 + 20, 6*20, 6*20, (50, 100, 220), thickness=2)
    c.draw_grid(250, 80, 10, 10, 16, grid_color=(150, 150, 150),
                fill_colors={(r, col): (255, 255, 255) if (r<2 or r>7 or col<2 or col>7) else (210, 230, 255)
                             for r in range(10) for col in range(10)})
    c.stroke_rect(250 + 2*16, 80 + 2*16, 6*16, 6*16, (50, 100, 220), thickness=2)
    c.draw_grid(460, 80, 4, 4, 22, grid_color=(160, 160, 160),
                fill_colors={(0,0): (255,200,200), (0,1): (255,200,200), (1,0): (255,200,200), (1,1): (255,200,200),
                             (0,2): (200,255,200), (0,3): (200,255,200), (1,2): (200,255,200), (1,3): (200,255,200),
                             (2,0): (200,200,255), (2,1): (200,200,255), (3,0): (200,200,255), (3,1): (200,200,255),
                             (2,2): (255,255,200), (2,3): (255,255,200), (3,2): (255,255,200), (3,3): (255,255,200)})
    c.draw_line(560, 124, 600, 124, (80, 80, 80), thickness=2)
    c.draw_grid(610, 95, 2, 2, 30, grid_color=(100, 100, 100),
                fill_colors={(0,0): (255,160,160), (0,1): (160,255,160), (1,0): (160,160,255), (1,1): (255,255,160)})
    c.draw_grid(460, 230, 4, 4, 22, grid_color=(160, 160, 160),
                fill_colors={(r, col): (220, 240, 255) for r in range(4) for col in range(4)})
    c.draw_line(560, 274, 600, 274, (80, 80, 80), thickness=2)
    c.draw_grid(610, 245, 2, 2, 30, grid_color=(100, 100, 100),
                fill_colors={(r, col): (180, 220, 255) for r in range(2) for col in range(2)})
    c.draw_grid(460, 380, 4, 4, 22, grid_color=(160, 160, 160),
                fill_colors={(r, col): (255, 235, 210) for r in range(4) for col in range(4)})
    c.draw_line(560, 424, 600, 424, (80, 80, 80), thickness=2)
    c.draw_grid(610, 395, 2, 2, 30, grid_color=(100, 100, 100),
                fill_colors={(r, col): (255, 210, 160) for r in range(2) for col in range(2)})
    c.fill_rect(710, 90, 190, 70, (255, 255, 255))
    c.stroke_rect(710, 90, 190, 70, (220, 80, 80), thickness=1)
    c.fill_rect(710, 240, 190, 70, (255, 255, 255))
    c.stroke_rect(710, 240, 190, 70, (80, 140, 220), thickness=1)
    c.fill_rect(710, 390, 190, 70, (255, 255, 255))
    c.stroke_rect(710, 390, 190, 70, (220, 140, 50), thickness=1)
    c.save_png(OUT / "padding_and_pooling.png")

def generate_fig4():
    w, h = 960, 460
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 880, 45, (235, 240, 255))
    c.stroke_rect(40, 15, 880, 45, (60, 90, 180), thickness=2)
    c.fill_rect(60, 140, 90, 90, (180, 140, 100))
    c.stroke_rect(60, 140, 90, 90, (50, 50, 50), thickness=2)
    c.draw_line(155, 185, 185, 185, (80, 80, 80), thickness=2)
    for i in range(4):
        c.fill_rect(190 + i*6, 125 - i*6, 50, 110, (100 + i*15, 150 + i*15, 240))
        c.stroke_rect(190 + i*6, 125 - i*6, 50, 110, (40, 80, 180), thickness=1)
    c.draw_line(275, 185, 305, 185, (80, 80, 80), thickness=2)
    for i in range(5):
        c.fill_rect(310 + i*6, 140 - i*6, 45, 80, (255, 180 + i*10, 80 + i*10))
        c.stroke_rect(310 + i*6, 140 - i*6, 45, 80, (200, 120, 20), thickness=1)
    c.draw_line(395, 185, 425, 185, (80, 80, 80), thickness=2)
    for i in range(6):
        c.fill_rect(430 + i*6, 155 - i*6, 35, 55, (100 + i*10, 220, 120 + i*10))
        c.stroke_rect(430 + i*6, 155 - i*6, 35, 55, (40, 150, 60), thickness=1)
    c.draw_line(510, 185, 545, 185, (80, 80, 80), thickness=2)
    c.fill_rect(550, 110, 16, 160, (50, 180, 120))
    c.stroke_rect(550, 110, 16, 160, (20, 100, 60), thickness=2)
    c.draw_line(575, 185, 605, 185, (80, 80, 80), thickness=2)
    for y_pos in range(120, 270, 22):
        c.fill_rect(615, y_pos, 14, 14, (70, 130, 220))
        c.stroke_rect(615, y_pos, 14, 14, (20, 60, 140), thickness=1)
    c.stroke_rect(605, 105, 34, 185, (220, 40, 40), thickness=2)
    c.draw_line(645, 185, 695, 185, (80, 80, 80), thickness=2)
    for idx, y_pos in enumerate([140, 175, 210, 245]):
        c.fill_rect(705, y_pos, 80, 24, (220, 240, 220))
        c.stroke_rect(705, y_pos, 80, 24, (40, 140, 50), thickness=1)
    c.fill_rect(60, 340, 460, 45, (240, 245, 255))
    c.stroke_rect(60, 340, 460, 45, (100, 120, 180), thickness=1)
    c.fill_rect(540, 340, 360, 45, (255, 240, 240))
    c.stroke_rect(540, 340, 360, 45, (200, 80, 80), thickness=1)
    c.save_png(OUT / "cnn_feature_vector_pipeline.png")

def generate_fig5():
    w, h = 940, 500
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 860, 45, (255, 235, 235))
    c.stroke_rect(40, 15, 860, 45, (200, 50, 50), thickness=2)
    c.draw_grid(60, 100, 7, 7, 18, grid_color=(180, 180, 180),
                fill_colors={(r, col): (255, 220, 180) for r in range(7) for col in range(7)})
    c.stroke_rect(60 + 2*18, 100 + 2*18, 3*18, 3*18, (220, 50, 50), thickness=2)
    c.draw_line(200, 160, 240, 160, (80, 80, 80), thickness=2)
    c.draw_grid(250, 118, 5, 5, 18, grid_color=(180, 180, 180),
                fill_colors={(r, col): (255, 240, 200) for r in range(5) for col in range(5)})
    c.stroke_rect(250 + 1*18, 118 + 1*18, 3*18, 3*18, (50, 150, 50), thickness=2)
    c.draw_line(350, 160, 390, 160, (80, 80, 80), thickness=2)
    c.draw_grid(400, 136, 3, 3, 18, grid_color=(180, 180, 180),
                fill_colors={(r, col): (200, 240, 255) for r in range(3) for col in range(3)})
    c.stroke_rect(400 + 1*18, 136 + 1*18, 18, 18, (50, 50, 200), thickness=3)
    c.fill_rect(60, 260, 420, 200, (255, 255, 255))
    c.stroke_rect(60, 260, 420, 200, (200, 200, 200), thickness=1)
    plot_x0, plot_y0, plot_w, plot_h = 540, 120, 340, 320
    c.fill_rect(plot_x0, plot_y0, plot_w, plot_h, (255, 255, 255))
    c.stroke_rect(plot_x0, plot_y0, plot_w, plot_h, (180, 180, 180), thickness=1)
    c.draw_line(plot_x0 + 40, plot_y0 + plot_h - 40, plot_x0 + plot_w - 20, plot_y0 + plot_h - 40, (50, 50, 50), thickness=2)
    c.draw_line(plot_x0 + 40, plot_y0 + plot_h - 40, plot_x0 + 40, plot_y0 + 30, (50, 50, 50), thickness=2)
    points = []
    num_pts = 10
    for i in range(num_pts):
        px = plot_x0 + 50 + int(i * (plot_w - 90) / (num_pts - 1))
        norm_i = i / (num_pts - 1)
        py = (plot_y0 + plot_h - 45) - int((norm_i ** 3.2) * (plot_h - 90))
        points.append((px, py))
    for i in range(len(points) - 1):
        c.draw_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], (220, 40, 40), thickness=3)
    for px, py in points:
        c.fill_rect(px - 5, py - 5, 10, 10, (220, 40, 40))
        c.stroke_rect(px - 5, py - 5, 10, 10, (120, 10, 10), thickness=1)
    c.fill_rect(plot_x0 + 45, plot_y0 + plot_h - 80, 120, 35, (255, 230, 230))
    c.stroke_rect(plot_x0 + 45, plot_y0 + plot_h - 80, 120, 35, (220, 50, 50), thickness=1)
    c.save_png(OUT / "receptive_field_vs_vanishing_gradient.png")

def generate_fig6():
    w, h = 940, 520
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 860, 45, (245, 235, 255))
    c.stroke_rect(40, 15, 860, 45, (140, 60, 200), thickness=2)
    c.fill_rect(50, 80, 400, 400, (255, 255, 255))
    c.stroke_rect(50, 80, 400, 400, (200, 200, 200), thickness=1)
    c.fill_rect(70, 110, 160, 140, (230, 240, 230))
    c.stroke_rect(70, 110, 160, 140, (100, 100, 100), thickness=1)
    c.fill_rect(70, 210, 160, 40, (80, 80, 80))
    c.fill_rect(100, 185, 60, 25, (40, 100, 200))
    c.stroke_rect(95, 180, 70, 35, (220, 40, 40), thickness=2)
    c.fill_rect(260, 110, 160, 140, (240, 235, 230))
    c.stroke_rect(260, 110, 160, 140, (100, 100, 100), thickness=1)
    c.fill_rect(290, 160, 100, 90, (140, 140, 150))
    c.fill_rect(310, 135, 60, 25, (40, 100, 200))
    c.stroke_rect(305, 130, 70, 35, (220, 40, 40), thickness=2)
    c.fill_rect(70, 280, 360, 180, (245, 245, 255))
    c.stroke_rect(70, 280, 360, 180, (150, 150, 220), thickness=1)
    c.fill_rect(480, 80, 420, 400, (255, 255, 255))
    c.stroke_rect(480, 80, 420, 400, (200, 200, 200), thickness=1)
    c.fill_rect(500, 100, 380, 60, (255, 255, 230))
    c.stroke_rect(500, 100, 380, 60, (200, 180, 50), thickness=1)
    c.draw_line(530, 130, 660, 110, (220, 50, 50), thickness=2)
    c.draw_line(660, 110, 800, 130, (220, 50, 50), thickness=2)
    c.fill_rect(500, 180, 380, 190, (220, 235, 245))
    c.stroke_rect(500, 180, 380, 190, (100, 140, 180), thickness=2)
    c.fill_rect(500, 330, 380, 40, (40, 40, 45))
    c.draw_line(500, 330, 680, 230, (180, 180, 180), thickness=2)
    c.draw_line(880, 330, 700, 230, (180, 180, 180), thickness=2)
    c.fill_rect(560, 250, 22, 55, (30, 30, 30))
    c.stroke_rect(552, 242, 38, 68, (220, 40, 40), thickness=2)
    c.fill_rect(770, 200, 20, 45, (20, 20, 20))
    c.fill_rect(776, 225, 8, 8, (40, 220, 60))
    c.stroke_rect(762, 192, 36, 58, (220, 40, 40), thickness=2)
    c.draw_line(575, 242, 762, 210, (220, 40, 40), thickness=3)
    c.fill_rect(500, 390, 380, 75, (240, 250, 240))
    c.stroke_rect(500, 390, 380, 75, (60, 160, 80), thickness=1)
    c.save_png(OUT / "cnn_limitations_and_long_range_dependency.png")

def generate_fig7():
    w, h = 980, 560
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 900, 50, (230, 245, 255))
    c.stroke_rect(40, 15, 900, 50, (30, 100, 200), thickness=2)
    c.draw_grid(60, 410, 3, 3, 32, grid_color=(40, 40, 40),
                fill_colors={(r, col): (180 + (r*25)%70, 200 + (col*20)%55, 230) for r in range(3) for col in range(3)})
    c.stroke_rect(60, 410, 96, 96, (20, 20, 20), thickness=2)
    c.draw_line(108, 400, 108, 365, (80, 80, 80), thickness=2)
    c.fill_rect(60, 320, 480, 36, (255, 215, 215))
    c.stroke_rect(60, 320, 480, 36, (200, 80, 80), thickness=2)
    c.fill_rect(60, 240, 36, 45, (255, 220, 120))
    c.stroke_rect(60, 240, 36, 45, (200, 120, 20), thickness=2)
    for i in range(1, 10):
        c.fill_rect(60 + i*48, 240, 36, 45, (210, 230, 255))
        c.stroke_rect(60 + i*48, 240, 36, 45, (50, 100, 200), thickness=1)
    for i in range(10):
        c.fill_rect(73 + i*48, 210, 12, 12, (230, 230, 230))
        c.stroke_rect(73 + i*48, 210, 12, 12, (80, 80, 80), thickness=1)
        c.draw_line(79 + i*48, 222, 79 + i*48, 238, (80, 80, 80), thickness=1)
    c.fill_rect(50, 110, 520, 85, (225, 225, 230))
    c.stroke_rect(50, 110, 520, 85, (80, 80, 90), thickness=2)
    c.draw_line(78, 105, 78, 80, (200, 50, 50), thickness=3)
    c.draw_line(78, 80, 140, 80, (200, 50, 50), thickness=3)
    c.fill_rect(145, 68, 80, 26, (255, 220, 160))
    c.stroke_rect(145, 68, 80, 26, (200, 100, 20), thickness=2)
    c.draw_line(230, 81, 260, 81, (50, 50, 50), thickness=2)
    c.fill_rect(265, 65, 90, 32, (220, 255, 220))
    c.stroke_rect(265, 65, 90, 32, (40, 160, 40), thickness=2)
    c.fill_rect(620, 80, 320, 450, (255, 255, 255))
    c.stroke_rect(620, 80, 320, 450, (100, 100, 100), thickness=2)
    c.fill_rect(700, 480, 160, 32, (255, 215, 215))
    c.stroke_rect(700, 480, 160, 32, (200, 80, 80), thickness=1)
    c.fill_rect(710, 420, 140, 30, (255, 245, 180))
    c.stroke_rect(710, 420, 140, 30, (180, 160, 20), thickness=1)
    c.fill_rect(690, 350, 180, 40, (180, 240, 180))
    c.stroke_rect(690, 350, 180, 40, (40, 160, 40), thickness=2)
    c.fill_rect(770, 300, 22, 22, (240, 240, 240))
    c.stroke_rect(770, 300, 22, 22, (50, 50, 50), thickness=2)
    c.draw_line(650, 460, 650, 311, (80, 80, 80), thickness=2)
    c.draw_line(650, 311, 768, 311, (80, 80, 80), thickness=2)
    c.fill_rect(710, 240, 140, 30, (255, 245, 180))
    c.stroke_rect(710, 240, 140, 30, (180, 160, 20), thickness=1)
    c.fill_rect(710, 170, 140, 40, (180, 210, 255))
    c.stroke_rect(710, 170, 140, 40, (40, 90, 190), thickness=2)
    c.fill_rect(770, 120, 22, 22, (240, 240, 240))
    c.stroke_rect(770, 120, 22, 22, (50, 50, 50), thickness=2)
    c.draw_line(665, 280, 665, 131, (80, 80, 80), thickness=2)
    c.draw_line(665, 131, 768, 131, (80, 80, 80), thickness=2)
    c.draw_line(780, 480, 780, 452, (80, 80, 80), thickness=2)
    c.draw_line(780, 420, 780, 392, (80, 80, 80), thickness=2)
    c.draw_line(780, 350, 780, 324, (80, 80, 80), thickness=2)
    c.draw_line(780, 298, 780, 272, (80, 80, 80), thickness=2)
    c.draw_line(780, 240, 780, 212, (80, 80, 80), thickness=2)
    c.draw_line(780, 170, 780, 144, (80, 80, 80), thickness=2)
    c.draw_line(780, 118, 780, 95, (80, 80, 80), thickness=2)
    c.save_png(OUT / "vision_transformer_architecture.png")

# ---------------------------------------------------------------------------
# Day 2 Figures (8 to 13)
# ---------------------------------------------------------------------------

def generate_fig8():
    # Tokenizer Paradigms & Byte Pair Encoding (BPE)
    w, h = 960, 520
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 880, 45, (230, 240, 255))
    c.stroke_rect(40, 15, 880, 45, (40, 100, 200), thickness=2)

    # 3 columns for Word-based, Sub-word (BPE), Character-based
    c.fill_rect(50, 80, 260, 220, (255, 235, 235))
    c.stroke_rect(50, 80, 260, 220, (200, 60, 60), thickness=2)
    # Checkmarks and crosses
    c.fill_rect(65, 110, 15, 15, (50, 160, 50))
    c.fill_rect(65, 150, 15, 15, (220, 50, 50))
    c.fill_rect(65, 190, 15, 15, (220, 50, 50))
    c.fill_rect(65, 230, 15, 15, (220, 50, 50))

    c.fill_rect(330, 80, 300, 220, (235, 255, 235))
    c.stroke_rect(330, 80, 300, 220, (40, 160, 60), thickness=3) # Highlight BPE!
    c.fill_rect(345, 110, 15, 15, (50, 160, 50))
    c.fill_rect(345, 150, 15, 15, (50, 160, 50))
    c.fill_rect(345, 190, 15, 15, (50, 160, 50))
    c.fill_rect(345, 230, 15, 15, (50, 160, 50))

    c.fill_rect(650, 80, 260, 220, (255, 245, 230))
    c.stroke_rect(650, 80, 260, 220, (220, 140, 30), thickness=2)
    c.fill_rect(665, 110, 15, 15, (50, 160, 50))
    c.fill_rect(665, 150, 15, 15, (50, 160, 50))
    c.fill_rect(665, 190, 15, 15, (220, 50, 50))
    c.fill_rect(665, 230, 15, 15, (220, 50, 50))

    # Lower section: BPE 4-Step Merge Walkthrough Box
    c.fill_rect(50, 320, 860, 180, (255, 255, 255))
    c.stroke_rect(50, 320, 860, 180, (180, 180, 180), thickness=2)

    # 4 Steps horizontal cards
    for idx, (label_bg, pair_txt) in enumerate([
        ((255, 240, 200), (220, 50, 50)),
        ((230, 245, 255), (40, 100, 200)),
        ((240, 230, 255), (140, 60, 200)),
        ((230, 255, 230), (40, 160, 60))
    ]):
        cx = 70 + idx * 205
        c.fill_rect(cx, 340, 195, 140, label_bg)
        c.stroke_rect(cx, 340, 195, 140, pair_txt, thickness=2)
        # Inner token sequence bar
        c.fill_rect(cx + 10, 400, 175, 30, (255, 255, 255))
        c.stroke_rect(cx + 10, 400, 175, 30, (100, 100, 100), thickness=1)

    c.save_png(OUT / "tokenizer_paradigms_bpe.png")

def generate_fig9():
    # Token Embedding in 768-D & Position Embedding Addition
    w, h = 940, 540
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 860, 45, (240, 240, 255))
    c.stroke_rect(40, 15, 860, 45, (70, 70, 200), thickness=2)

    # Left: Semantic Subspace Plot (768th dim vs 1st dim)
    c.fill_rect(60, 80, 360, 230, (255, 255, 255))
    c.stroke_rect(60, 80, 360, 230, (180, 180, 180), thickness=1)
    # Axes
    c.draw_line(80, 280, 380, 280, (40, 40, 40), thickness=2) # 1st dim
    c.draw_line(80, 280, 80, 100, (40, 40, 40), thickness=2)  # 768th dim
    # Animal cluster (ellipse fill)
    c.fill_rect(120, 110, 90, 80, (255, 220, 220))
    c.stroke_rect(120, 110, 90, 80, (220, 60, 60), thickness=1)
    # Fruit cluster
    c.fill_rect(240, 180, 90, 70, (220, 220, 255))
    c.stroke_rect(240, 180, 90, 70, (60, 60, 220), thickness=1)
    # Vectors
    c.draw_line(80, 280, 150, 130, (220, 40, 40), thickness=3) # Cat
    c.draw_line(80, 280, 180, 150, (220, 40, 40), thickness=3) # Dog
    c.draw_line(80, 280, 270, 200, (40, 60, 220), thickness=3) # Apple
    c.draw_line(80, 280, 300, 220, (40, 60, 220), thickness=3) # Orange

    # Right: Position Embedding Modifying Vector Direction
    c.fill_rect(460, 80, 440, 230, (255, 255, 255))
    c.stroke_rect(460, 80, 440, 230, (180, 180, 180), thickness=1)
    # Axes 1
    c.draw_line(490, 280, 650, 280, (40, 40, 40), thickness=2)
    c.draw_line(490, 280, 490, 120, (40, 40, 40), thickness=2)
    # Dog 1 (pos 2)
    c.draw_line(490, 280, 580, 150, (220, 40, 40), thickness=3)
    # Axes 2
    c.draw_line(700, 280, 860, 280, (40, 40, 40), thickness=2)
    c.draw_line(700, 280, 700, 120, (40, 40, 40), thickness=2)
    # Dog 2 (pos 5 - modified direction)
    c.draw_line(700, 280, 810, 180, (140, 40, 200), thickness=3)

    # Lower Section: Element-Wise Vector Addition Formula & Cells
    c.fill_rect(60, 330, 840, 180, (255, 255, 255))
    c.stroke_rect(60, 330, 840, 180, (150, 150, 200), thickness=2)

    # Token embedding row (green)
    c.draw_grid(100, 360, 1, 10, 26, grid_color=(40, 120, 40),
                fill_colors={(0, i): (200 + (i*15)%45, 255, 200 + (i*15)%45) for i in range(10)})
    # Plus sign
    c.fill_rect(380, 368, 14, 14, (60, 60, 60))
    # Position embedding row (pink)
    c.draw_grid(420, 360, 1, 10, 26, grid_color=(160, 40, 40),
                fill_colors={(0, i): (255, 210 + (i*12)%40, 210 + (i*12)%40) for i in range(10)})
    # Equals sign
    c.draw_line(700, 370, 725, 370, (40, 40, 40), thickness=3)
    c.draw_line(700, 378, 725, 378, (40, 40, 40), thickness=3)
    # Input embedding row (blue)
    c.draw_grid(750, 360, 1, 10, 26, grid_color=(40, 60, 180),
                fill_colors={(0, i): (210 + (i*10)%40, 225 + (i*10)%25, 255) for i in range(10)})

    # Word order sensitivity demonstration box
    c.fill_rect(100, 420, 750, 70, (245, 245, 255))
    c.stroke_rect(100, 420, 750, 70, (120, 120, 220), thickness=1)

    c.save_png(OUT / "token_and_position_embeddings.png")

def generate_fig10():
    # Complete 3-Stage LLM Pipeline & Transformer Block Series
    w, h = 980, 560
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 900, 45, (230, 245, 240))
    c.stroke_rect(40, 15, 900, 45, (40, 150, 120), thickness=2)

    # Stage 1: Input [Prompt]
    c.fill_rect(50, 80, 240, 340, (255, 255, 255))
    c.stroke_rect(50, 80, 240, 340, (100, 100, 100), thickness=2)
    # Inner modules
    c.fill_rect(70, 120, 200, 45, (235, 235, 245)) # Tokenized text
    c.stroke_rect(70, 120, 200, 45, (80, 80, 160), thickness=1)
    c.draw_line(170, 165, 170, 195, (80, 80, 80), thickness=2)
    c.fill_rect(70, 195, 200, 45, (220, 245, 220)) # Token embedding
    c.stroke_rect(70, 195, 200, 45, (40, 140, 60), thickness=1)
    c.draw_line(170, 240, 170, 270, (80, 80, 80), thickness=2)
    c.fill_rect(70, 270, 200, 45, (255, 230, 230)) # Position embedding
    c.stroke_rect(70, 270, 200, 45, (180, 60, 60), thickness=1)

    # Arrow Stage 1 -> Stage 2
    c.draw_line(270, 292, 330, 292, (80, 60, 220), thickness=3)
    c.draw_line(330, 292, 330, 450, (80, 60, 220), thickness=3)
    c.draw_line(330, 450, 370, 450, (80, 60, 220), thickness=3)

    # Stage 2: Transformer Block (Pre-LN)
    c.fill_rect(350, 80, 280, 420, (240, 240, 245))
    c.stroke_rect(350, 80, 280, 420, (60, 60, 80), thickness=2)
    # Modules inside Block
    c.fill_rect(390, 430, 180, 30, (200, 240, 240)) # LN 1
    c.stroke_rect(390, 430, 180, 30, (30, 140, 140), thickness=1)
    c.fill_rect(380, 370, 200, 38, (255, 220, 160)) # MHA
    c.stroke_rect(380, 370, 200, 38, (200, 100, 20), thickness=2)
    c.fill_rect(400, 320, 160, 26, (255, 200, 200)) # Dropout 1
    c.stroke_rect(400, 320, 160, 26, (180, 60, 60), thickness=1)
    c.fill_rect(470, 275, 20, 20, (240, 240, 240)) # (+)
    c.stroke_rect(470, 275, 20, 20, (50, 50, 50), thickness=2)
    # Skip line 1
    c.draw_line(370, 445, 370, 285, (80, 80, 80), thickness=2)
    c.draw_line(370, 285, 470, 285, (80, 80, 80), thickness=2)

    c.fill_rect(390, 225, 180, 30, (200, 240, 240)) # LN 2
    c.stroke_rect(390, 225, 180, 30, (30, 140, 140), thickness=1)
    c.fill_rect(380, 165, 200, 38, (220, 220, 245)) # FFN
    c.stroke_rect(380, 165, 200, 38, (60, 60, 160), thickness=2)
    c.fill_rect(400, 120, 160, 26, (255, 200, 200)) # Dropout 2
    c.stroke_rect(400, 120, 160, 26, (180, 60, 60), thickness=1)
    c.fill_rect(470, 90, 20, 20, (240, 240, 240))  # (+)
    c.stroke_rect(470, 90, 20, 20, (50, 50, 50), thickness=2)
    # Skip line 2
    c.draw_line(370, 260, 370, 100, (80, 80, 80), thickness=2)
    c.draw_line(370, 100, 470, 100, (80, 80, 80), thickness=2)

    # Arrow Stage 2 -> Stage 3
    c.draw_line(490, 100, 680, 100, (80, 60, 220), thickness=3)

    # Stage 3: Output [Next Word]
    c.fill_rect(680, 80, 250, 340, (255, 255, 255))
    c.stroke_rect(680, 80, 250, 340, (100, 100, 100), thickness=2)
    c.fill_rect(705, 120, 200, 40, (200, 240, 240)) # Final LN
    c.stroke_rect(705, 120, 200, 40, (30, 140, 140), thickness=1)
    c.draw_line(805, 160, 805, 195, (80, 80, 80), thickness=2)
    c.fill_rect(705, 195, 200, 45, (255, 230, 200)) # Output Projection Head
    c.stroke_rect(705, 195, 200, 45, (200, 120, 20), thickness=2)
    c.draw_line(805, 240, 805, 275, (80, 80, 80), thickness=2)
    c.fill_rect(705, 275, 200, 45, (220, 255, 220)) # Next word "mat"
    c.stroke_rect(705, 275, 200, 45, (40, 160, 60), thickness=2)

    # Bottom Banner: 12 Blocks in Series Indicator
    c.fill_rect(50, 440, 280, 60, (255, 245, 220))
    c.stroke_rect(50, 440, 280, 60, (200, 140, 30), thickness=1)

    c.fill_rect(680, 440, 250, 60, (240, 255, 240))
    c.stroke_rect(680, 440, 250, 60, (40, 160, 60), thickness=1)

    c.save_png(OUT / "transformer_block_architecture.png")

def generate_fig11():
    # Layer Normalization (with exact numerical example) & Dropout Scaling
    w, h = 940, 520
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 860, 45, (235, 250, 245))
    c.stroke_rect(40, 15, 860, 45, (30, 150, 120), thickness=2)

    # Left: Layer Normalization Mathematical Computation Box
    c.fill_rect(50, 80, 420, 410, (255, 255, 255))
    c.stroke_rect(50, 80, 420, 410, (180, 180, 180), thickness=2)

    # Raw input vector x = [2.0, -1.0, 3.0, 0.0]
    c.draw_grid(80, 120, 1, 4, 45, grid_color=(180, 40, 40),
                fill_colors={(0, 0): (255, 200, 200), (0, 1): (255, 180, 180), (0, 2): (255, 220, 220), (0, 3): (240, 240, 240)})
    # Mean & Variance steps card
    c.fill_rect(80, 190, 360, 95, (245, 245, 255))
    c.stroke_rect(80, 190, 360, 95, (100, 100, 200), thickness=1)
    # Output normalized vector x_hat = [0.63, -1.26, 1.26, -0.63]
    c.draw_grid(80, 310, 1, 4, 45, grid_color=(40, 140, 40),
                fill_colors={(0, 0): (210, 255, 210), (0, 1): (180, 240, 180), (0, 2): (220, 255, 220), (0, 3): (200, 240, 200)})
    # Learnable Gamma and Beta card
    c.fill_rect(80, 385, 360, 85, (255, 255, 230))
    c.stroke_rect(80, 385, 360, 85, (200, 160, 30), thickness=1)

    # Right: Dropout Regularization (Active vs Lazy Neurons & Scaling)
    c.fill_rect(500, 80, 400, 410, (255, 255, 255))
    c.stroke_rect(500, 80, 400, 410, (180, 180, 180), thickness=2)

    # Layer 1: Fully connected without dropout
    for y_idx in range(4):
        c.fill_rect(540, 130 + y_idx * 35, 20, 20, (100, 200, 100))
        c.stroke_rect(540, 130 + y_idx * 35, 20, 20, (40, 120, 40), thickness=2)
        c.fill_rect(620, 130 + y_idx * 35, 20, 20, (100, 200, 100))
        c.stroke_rect(620, 130 + y_idx * 35, 20, 20, (40, 120, 40), thickness=2)
        c.draw_line(560, 140 + y_idx * 35, 620, 140 + y_idx * 35, (160, 160, 160), thickness=1)

    # Layer 2: With Dropout (Masked lazy neurons shown in red)
    for y_idx in range(4):
        c.fill_rect(730, 130 + y_idx * 35, 20, 20, (100, 200, 100) if y_idx%2==0 else (240, 70, 70))
        c.stroke_rect(730, 130 + y_idx * 35, 20, 20, (40, 120, 40) if y_idx%2==0 else (180, 20, 20), thickness=2)
        c.fill_rect(810, 130 + y_idx * 35, 20, 20, (100, 200, 100) if y_idx%2!=0 else (240, 70, 70))
        c.stroke_rect(810, 130 + y_idx * 35, 20, 20, (40, 120, 40) if y_idx%2!=0 else (180, 20, 20), thickness=2)

    # Dropout inverted scaling explanation card
    c.fill_rect(520, 310, 360, 160, (255, 240, 240))
    c.stroke_rect(520, 310, 360, 160, (220, 80, 80), thickness=1)

    c.save_png(OUT / "layer_norm_and_dropout.png")

def generate_fig12():
    # FFN 4x Expansion & Skip Connections Mitigating Vanishing Gradients
    w, h = 960, 520
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 880, 45, (245, 235, 255))
    c.stroke_rect(40, 15, 880, 45, (120, 60, 200), thickness=2)

    # Left: Feed-Forward Network Dimension Expansion (768 -> 3072 -> 768)
    c.fill_rect(50, 80, 430, 410, (255, 255, 255))
    c.stroke_rect(50, 80, 430, 410, (180, 180, 180), thickness=2)

    # Input 768 bar
    c.fill_rect(80, 200, 35, 160, (100, 160, 255))
    c.stroke_rect(80, 200, 35, 160, (40, 80, 180), thickness=2)

    # Arrow & W1
    c.draw_line(125, 280, 195, 280, (60, 60, 60), thickness=3)

    # Expanded 3072 (4x) bar
    c.fill_rect(205, 110, 50, 340, (255, 180, 100))
    c.stroke_rect(205, 110, 50, 340, (200, 100, 20), thickness=3)

    # Arrow & W2
    c.draw_line(265, 280, 335, 280, (60, 60, 60), thickness=3)

    # Output 768 bar
    c.fill_rect(345, 200, 35, 160, (100, 160, 255))
    c.stroke_rect(345, 200, 35, 160, (40, 80, 180), thickness=2)

    # Parameter distribution summary card (66% in FFN!)
    c.fill_rect(80, 385, 300, 85, (255, 250, 220))
    c.stroke_rect(80, 385, 300, 85, (200, 140, 20), thickness=1)

    # Right: Skip Connection Gradient Flow Comparison
    c.fill_rect(510, 80, 410, 410, (255, 255, 255))
    c.stroke_rect(510, 80, 410, 410, (180, 180, 180), thickness=2)

    # Top right: Without skip (gradient decays to 0)
    c.fill_rect(530, 100, 370, 160, (255, 235, 235))
    c.stroke_rect(530, 100, 370, 160, (220, 60, 60), thickness=1)
    # Decay curve
    c.draw_line(560, 240, 860, 240, (40, 40, 40), thickness=2)
    c.draw_line(560, 240, 560, 120, (40, 40, 40), thickness=2)
    for i in range(8):
        x1 = 570 + i*35
        y1 = 235 - int(((i/7)**3.5) * 90)
        x2 = 570 + (i+1)*35
        y2 = 235 - int((((i+1)/7)**3.5) * 90)
        if i < 7: c.draw_line(x1, y1, x2, y2, (220, 40, 40), thickness=3)

    # Bottom right: With skip connection (gradient sustained)
    c.fill_rect(530, 280, 370, 190, (235, 255, 235))
    c.stroke_rect(530, 280, 370, 190, (40, 160, 60), thickness=1)
    c.draw_line(560, 440, 860, 440, (40, 40, 40), thickness=2)
    c.draw_line(560, 440, 560, 300, (40, 40, 40), thickness=2)
    for i in range(8):
        x1 = 570 + i*35
        y1 = 370 - int((i/7) * 45)
        x2 = 570 + (i+1)*35
        y2 = 370 - int(((i+1)/7) * 45)
        if i < 7: c.draw_line(x1, y1, x2, y2, (40, 160, 60), thickness=3)

    c.save_png(OUT / "ffn_expansion_and_skip_connections.png")

def generate_fig13():
    # Stage 3 Output Layer NN, Logits Projection & Self-Supervised Targets
    w, h = 960, 540
    c = Canvas(w, h, bg=(248, 249, 250))
    c.fill_rect(40, 15, 880, 45, (230, 240, 255))
    c.stroke_rect(40, 15, 880, 45, (30, 90, 200), thickness=2)

    # Left: Output Projection Layer (768 -> 50,257)
    c.fill_rect(50, 80, 420, 430, (255, 255, 255))
    c.stroke_rect(50, 80, 420, 430, (180, 180, 180), thickness=2)

    # 5 Tokens input
    for idx, name in enumerate(["The", "cat", "sat", "on", "the"]):
        cy = 110 + idx * 45
        c.fill_rect(70, cy, 60, 30, (235, 235, 245))
        c.stroke_rect(70, cy, 60, 30, (80, 80, 160), thickness=1)
        c.fill_rect(140, cy + 5, 50, 20, (100, 160, 255))
        c.draw_line(195, cy + 15, 245, 220, (160, 160, 160), thickness=1)

    # Central Projection Layer Sphere
    c.fill_rect(245, 175, 90, 90, (180, 245, 200))
    c.stroke_rect(245, 175, 90, 90, (30, 160, 80), thickness=3)

    # Logits output vectors (50,257 dimension)
    for idx in range(5):
        cy = 110 + idx * 45
        c.draw_line(335, 220, 375, cy + 15, (160, 160, 160), thickness=1)
        c.draw_grid(375, cy + 5, 1, 8, 10, grid_color=(40, 40, 160),
                    fill_colors={(0, 3): (220, 40, 40)})

    # Logits to Softmax explanation card
    c.fill_rect(70, 360, 380, 130, (245, 245, 255))
    c.stroke_rect(70, 360, 380, 130, (120, 120, 220), thickness=1)

    # Right: Self-Supervised Next-Token Training Pairs
    c.fill_rect(490, 80, 430, 430, (255, 255, 255))
    c.stroke_rect(490, 80, 430, 430, (180, 180, 180), thickness=2)

    # Training pairs table rows
    pairs = [
        ("Input: 'The'", "Ground Truth: 'cat'", "Pred: 'boy'", (255, 220, 220)),
        ("Input: 'The cat'", "Ground Truth: 'sat'", "Pred: 'sits'", (255, 220, 220)),
        ("Input: 'The cat sat'", "Ground Truth: 'on'", "Pred: 'on'", (220, 255, 220)),
        ("Input: 'The cat sat on'", "Ground Truth: 'the'", "Pred: 'a'", (255, 220, 220)),
        ("Input: 'The cat sat on the'", "Ground Truth: 'mat'", "Pred: 'chair'", (255, 220, 220))
    ]
    for idx, (inp, gt, pred, bg) in enumerate(pairs):
        ry = 110 + idx * 60
        c.fill_rect(510, ry, 390, 50, bg)
        c.stroke_rect(510, ry, 390, 50, (140, 140, 140), thickness=1)

    # Loss computation explanation card
    c.fill_rect(510, 420, 390, 75, (255, 255, 230))
    c.stroke_rect(510, 420, 390, 75, (200, 160, 20), thickness=1)

    c.save_png(OUT / "output_projection_and_next_token.png")


def main():
    print("Generating deterministic figures for Vision Transformers Week 1...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    print("Day 1 figures generated.")
    generate_fig8()
    generate_fig9()
    generate_fig10()
    generate_fig11()
    generate_fig12()
    generate_fig13()
    print("Day 2 figures generated.")
    print("Saved all figures to:", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(f" - {p.name} ({p.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
