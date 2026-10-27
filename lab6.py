# -*- coding: utf-8 -*-
import math
import colorsys
import tkinter as tk
from tkinter import ttk, messagebox

EPS = 1e-9

# ЧАСТЬ 1. МАТЕМАТИЧЕСКОЕ ЯДРО

class Point:
    """Точка/вектор в однородных координатах (x, y, z, w)"""
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def copy(self):
        return Point(self.x, self.y, self.z, self.w)

    def __repr__(self):
        return "Point(%.3f, %.3f, %.3f, %.3f)" % (self.x, self.y, self.z, self.w)


# --- вспомогательные операции над векторами (w = 0) -------------------------

def vec_sub(a, b):
    return Point(a.x - b.x, a.y - b.y, a.z - b.z, 0.0)


def vec_dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def vec_cross(a, b):
    return Point(a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x, 0.0)


def vec_len(a):
    return math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)


def vec_norm(a):
    L = vec_len(a)
    if L < EPS:
        return Point(0.0, 0.0, 0.0, 0.0)
    return Point(a.x / L, a.y / L, a.z / L, 0.0)


class Matrix4:
    """Однородная матрица 4x4."""
    __slots__ = ("m",)

    def __init__(self, data=None):
        if data is None:
            self.m = [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
        else:
            self.m = [[float(v) for v in row] for row in data]

    # ---------------- базовые операции ----------------

    def __matmul__(self, other):
        """Произведение матриц: (A @ B) - сначала B, затем A"""
        a, b = self.m, other.m
        res = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            ai = a[i]
            ri = res[i]
            for k in range(4):
                aik = ai[k]
                if aik == 0.0:
                    continue
                bk = b[k]
                ri[0] += aik * bk[0]
                ri[1] += aik * bk[1]
                ri[2] += aik * bk[2]
                ri[3] += aik * bk[3]
        return Matrix4(res)

    def apply(self, p):
        """Умножение матрицы на точку (p - вектор-столбец)"""
        m = self.m
        return Point(m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3] * p.w,
                     m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3] * p.w,
                     m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3] * p.w,
                     m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3] * p.w)

    # ---------------- конструкторы аффинных преобразований ----------------

    @staticmethod
    def identity():
        return Matrix4()

    @staticmethod
    def translation(dx, dy, dz):
        """Смещение (перенос)."""
        return Matrix4([[1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

    @staticmethod
    def scaling(sx, sy, sz):
        """Масштабирование относительно начала координат"""
        return Matrix4([[sx, 0, 0, 0],
                        [0, sy, 0, 0],
                        [0, 0, sz, 0],
                        [0, 0, 0, 1]])

    @staticmethod
    def rotation_x(a):
        c, s = math.cos(a), math.sin(a)
        return Matrix4([[1, 0,  0, 0],
                        [0, c, -s, 0],
                        [0, s,  c, 0],
                        [0, 0,  0, 1]])

    @staticmethod
    def rotation_y(a):
        c, s = math.cos(a), math.sin(a)
        return Matrix4([[ c, 0, s, 0],
                        [ 0, 1, 0, 0],
                        [-s, 0, c, 0],
                        [ 0, 0, 0, 1]])

    @staticmethod
    def rotation_z(a):
        c, s = math.cos(a), math.sin(a)
        return Matrix4([[c, -s, 0, 0],
                        [s,  c, 0, 0],
                        [0,  0, 1, 0],
                        [0,  0, 0, 1]])

    @staticmethod
    def reflection(plane):
        """Отражение относительно координатной плоскости (проходящей через 0)"""
        if plane == "OXY":                 # z -> -z
            return Matrix4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if plane == "OXZ":                 # y -> -y
            return Matrix4([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if plane == "OYZ":                 # x -> -x
            return Matrix4([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return Matrix4()

    @staticmethod
    def rotation_about_line(p0, p1, angle):
        """Поворот на угол angle вокруг прямой, заданной двумя точками p0, p1"""
        d = vec_sub(p1, p0)
        L = vec_len(d)
        if L < EPS:
            return Matrix4.identity()
        a, b, c = d.x / L, d.y / L, d.z / L
        ca, sa = math.cos(angle), math.sin(angle)
        t = 1.0 - ca
        R = [[a * a * t + ca,     a * b * t - c * sa, a * c * t + b * sa, 0.0],
             [a * b * t + c * sa, b * b * t + ca,     b * c * t - a * sa, 0.0],
             [a * c * t - b * sa, b * c * t + a * sa, c * c * t + ca,     0.0],
             [0.0, 0.0, 0.0, 1.0]]
        return (Matrix4.translation(p0.x, p0.y, p0.z)
                @ Matrix4(R)
                @ Matrix4.translation(-p0.x, -p0.y, -p0.z))


class Polygon:
    """Грань многогранника: циклический список индексов вершин + цвет"""
    __slots__ = ("indices", "color")

    def __init__(self, indices, color=(180, 180, 220)):
        self.indices = list(indices)
        self.color = tuple(color)

    def normal(self, vertices):
        """Нормаль грани по методу Ньюэлла"""
        nx = ny = nz = 0.0
        n = len(self.indices)
        for i in range(n):
            p = vertices[self.indices[i]]
            q = vertices[self.indices[(i + 1) % n]]
            nx += (p.y - q.y) * (p.z + q.z)
            ny += (p.z - q.z) * (p.x + q.x)
            nz += (p.x - q.x) * (p.y + q.y)
        return vec_norm(Point(nx, ny, nz, 0.0))

    def triangulate(self):
        """Веерная триангуляция грани"""
        idx = self.indices
        return [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]


# ЧАСТЬ 2. МНОГОГРАННИКИ, ПРОЕКЦИИ И ПОПИКСЕЛЬНЫЙ РАСТЕРИЗАТОР

PHI = (1.0 + math.sqrt(5.0)) / 2.0          # золотое сечение


def _face_colors(n):
    """Разные оттенки для граней - чтобы форма читалась"""
    res = []
    for i in range(n):
        h = i / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.50, 1.0)
        res.append((int(r * 255), int(g * 255), int(b * 255)))
    return res


class Polyhedron:
    """Многогранник: вершины + грани (объекты Polygon)"""

    def __init__(self, name, vertices, faces):
        self.name = name
        self.base_vertices = [v.copy() for v in vertices]
        self.vertices = [v.copy() for v in vertices]
        self.faces = faces

    def reset(self):
        self.vertices = [v.copy() for v in self.base_vertices]

    def apply(self, matrix):
        """Применить матрицу 4x4 ко всем вершинам (однородное деление)"""
        new_vertices = []
        for v in self.vertices:
            p = matrix.apply(v)
            if abs(p.w) > EPS and p.w != 1.0:
                p = Point(p.x / p.w, p.y / p.w, p.z / p.w, 1.0)
            new_vertices.append(p)
        self.vertices = new_vertices

    def center(self):
        n = len(self.vertices)
        if n == 0:
            return Point()
        sx = sy = sz = 0.0
        for v in self.vertices:
            sx += v.x
            sy += v.y
            sz += v.z
        return Point(sx / n, sy / n, sz / n)

    def edges(self):
        """Уникальные рёбра (для каркасной отрисовки поверх заливки)"""
        seen = set()
        res = []
        for f in self.faces:
            idx = f.indices
            n = len(idx)
            for i in range(n):
                a, b = idx[i], idx[(i + 1) % n]
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    res.append(key)
        return res


def _make_solid(name, vertices, faces):
    """Нормирует многогранник по радиусу описанной сферы и раскрашивает грани"""
    r = max(vec_len(v) for v in vertices)
    s = 1.6 / r
    verts = [Point(v.x * s, v.y * s, v.z * s) for v in vertices]
    cols = _face_colors(len(faces))
    polys = [Polygon(f, cols[i]) for i, f in enumerate(faces)]
    return Polyhedron(name, verts, polys)


# ---- правильные многогранники ----------------------------------------------

def build_tetrahedron():
    v = [Point(1, 1, 1), Point(1, -1, -1), Point(-1, 1, -1), Point(-1, -1, 1)]
    f = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return _make_solid("Тетраэдр", v, f)


def build_hexahedron():
    v = [Point(-1, -1, -1), Point(1, -1, -1), Point(1, 1, -1), Point(-1, 1, -1),
         Point(-1, -1,  1), Point(1, -1,  1), Point(1, 1,  1), Point(-1, 1,  1)]
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return _make_solid("Гексаэдр", v, f)


def build_octahedron():
    v = [Point(1, 0, 0), Point(-1, 0, 0), Point(0, 1, 0),
         Point(0, -1, 0), Point(0, 0, 1), Point(0, 0, -1)]
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return _make_solid("Октаэдр", v, f)


def build_icosahedron():
    p = PHI
    v = [Point(0, 1, p), Point(0, 1, -p), Point(0, -1, p), Point(0, -1, -p),
         Point(1, p, 0), Point(1, -p, 0), Point(-1, p, 0), Point(-1, -p, 0),
         Point(p, 0, 1), Point(p, 0, -1), Point(-p, 0, 1), Point(-p, 0, -1)]
    f = [(0, 2, 8), (0, 8, 4), (0, 4, 6), (0, 6, 10), (0, 10, 2),
         (2, 8, 5), (2, 5, 7), (2, 7, 10),
         (5, 8, 9), (5, 9, 3), (5, 3, 7),
         (9, 8, 4), (9, 4, 1), (9, 1, 3),
         (1, 4, 6), (1, 6, 11), (1, 11, 3),
         (6, 10, 11), (10, 11, 7), (3, 11, 7)]
    return _make_solid("Икосаэдр", v, f)


def build_dodecahedron():
    p = PHI
    b = 1.0 / p
    v = [Point( 1,  1,  1), Point( 1,  1, -1), Point( 1, -1,  1), Point( 1, -1, -1),
         Point(-1,  1,  1), Point(-1,  1, -1), Point(-1, -1,  1), Point(-1, -1, -1),
         Point( 0,  b,  p), Point( 0,  b, -p), Point( 0, -b,  p), Point( 0, -b, -p),
         Point( b,  p,  0), Point( b, -p,  0), Point(-b,  p,  0), Point(-b, -p,  0),
         Point( p,  0,  b), Point( p,  0, -b), Point(-p,  0,  b), Point(-p,  0, -b)]
    f = [(0, 8, 10, 2, 16), (0, 16, 17, 1, 12), (0, 12, 14, 4, 8),
         (8, 4, 18, 6, 10), (2, 13, 15, 6, 10), (2, 16, 17, 3, 13),
         (6, 15, 7, 19, 18), (4, 14, 5, 19, 18), (14, 12, 1, 9, 5),
         (5, 9, 11, 7, 19), (1, 17, 3, 11, 9), (3, 13, 15, 7, 11)]
    return _make_solid("Додекаэдр", v, f)


# ---- проецирование ----------------------------------------------------------

class Projection:
    """Матрица перехода "мир -> экран" для двух режимов - перспективная и аксонометрическая"""

    def __init__(self, cam_dist=8.0, scale=130.0):
        self.cam_dist = cam_dist
        self.scale = scale
        self.mode = "perspective"

    def matrix(self):
        if self.mode == "perspective":
            d = self.cam_dist
            return Matrix4([[d, 0, 0, 0],
                            [0, d, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, d]])
        alpha = math.radians(-35.264389682754654)
        beta = math.radians(135.0)
        return Matrix4.rotation_x(alpha) @ Matrix4.rotation_y(beta)


class Renderer:
    """Попиксельный программный рендерер с z-буфером"""

    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.projection = Projection()
        self.bg = (16, 16, 24)
        self.buf = bytearray(bytes(self.bg) * (width * height))
        self.zbuf = [float("inf")] * (width * height)

        # освещение (модель Ламберта, плоское затенение)
        self.ambient = 0.30
        self.diffuse = 0.70
        self.light = vec_norm(Point(0.35, 0.55, -1.0, 0.0))

        self.edge_color = (250, 250, 250)
        self.edge_bias = 0.01

    # ---------------- служебное ----------------

    def clear(self):
        self.buf = bytearray(bytes(self.bg) * (self.W * self.H))
        self.zbuf = [float("inf")] * (self.W * self.H)

    def _put_pixel(self, idx, color):
        i3 = idx * 3
        self.buf[i3] = color[0]
        self.buf[i3 + 1] = color[1]
        self.buf[i3 + 2] = color[2]

    # ---------------- растеризация треугольника ----------------

    def _raster_triangle(self, p0, p1, p2, color):
        """Заливка треугольника методом барицентрических координат"""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        minx = int(math.floor(min(x0, x1, x2)))
        maxx = int(math.ceil(max(x0, x1, x2)))
        miny = int(math.floor(min(y0, y1, y2)))
        maxy = int(math.ceil(max(y0, y1, y2)))
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx > self.W - 1:
            maxx = self.W - 1
        if maxy > self.H - 1:
            maxy = self.H - 1
        if minx > maxx or miny > maxy:
            return

        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            return
        inv_den = 1.0 / den

        W = self.W
        buf = self.buf
        zb = self.zbuf
        r, g, b = color
        eps = -1e-6

        for py in range(miny, maxy + 1):
            fy = py + 0.5
            base = py * W
            for px in range(minx, maxx + 1):
                fx = px + 0.5
                l0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * inv_den
                if l0 < eps:
                    continue
                l1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * inv_den
                if l1 < eps:
                    continue
                l2 = 1.0 - l0 - l1
                if l2 < eps:
                    continue
                z = l0 * z0 + l1 * z1 + l2 * z2
                idx = base + px
                if z < zb[idx]:
                    zb[idx] = z
                    i3 = idx * 3
                    buf[i3] = r
                    buf[i3 + 1] = g
                    buf[i3 + 2] = b

    # ---------------- целочисленный алгоритм Брезенхема ----------------

    def _bresenham(self, p0, p1, color):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x0 = int(round(x0)); y0 = int(round(y0))
        x1 = int(round(x1)); y1 = int(round(y1))

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        steps = max(dx, dy)
        if steps == 0:
            steps = 1

        i = 0
        x, y = x0, y0
        W, H = self.W, self.H
        zb = self.zbuf
        bias = self.edge_bias

        while True:
            t = i / float(steps)
            z = z0 + (z1 - z0) * t
            if 0 <= x < W and 0 <= y < H:
                idx = y * W + x
                if z - bias <= zb[idx]:
                    zb[idx] = z - bias
                    self._put_pixel(idx, color)

            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            i += 1

    # ---------------- главный метод ----------------

    def render(self, poly):
        self.clear()
        Pm = self.projection.matrix()
        scale = self.projection.scale
        cx = self.W / 2.0
        cy = self.H / 2.0

        # 1) проецируем все вершины (матрица -> однородное деление -> экран)
        screen = []
        zmin, zmax = float("inf"), float("-inf")
        for v in poly.vertices:
            p = Pm.apply(v)
            w = p.w
            if w < 1e-3:                       # отсечение по ближней плоскости
                screen.append(None)
                continue
            x = p.x / w
            y = p.y / w
            z = p.z / w
            if z < zmin:
                zmin = z
            if z > zmax:
                zmax = z
            screen.append((cx + x * scale, cy - y * scale, z))

        if zmax > zmin:
            self.edge_bias = max(1e-5, (zmax - zmin) * 0.012)

        # 2) заливаем грани (с учётом освещения)
        for face in poly.faces:
            n = face.normal(poly.vertices)
            inten = self.ambient + self.diffuse * abs(vec_dot(n, self.light))
            if inten > 1.0:
                inten = 1.0
            col = (min(255, int(face.color[0] * inten)),
                   min(255, int(face.color[1] * inten)),
                   min(255, int(face.color[2] * inten)))
            for (a, b, c) in face.triangulate():
                pa, pb, pc = screen[a], screen[b], screen[c]
                if pa is None or pb is None or pc is None:
                    continue
                self._raster_triangle(pa, pb, pc, col)

        # 3) поверх заливки рисуем рёбра каркаса
        for (a, b) in poly.edges():
            pa, pb = screen[a], screen[b]
            if pa is None or pb is None:
                continue
            self._bresenham(pa, pb, self.edge_color)


# ЧАСТЬ 3. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС И СБОРКА

SOLIDS = [
    ("Тетраэдр",       build_tetrahedron),
    ("Гексаэдр (куб)", build_hexahedron),
    ("Октаэдр",        build_octahedron),
    ("Икосаэдр",       build_icosahedron),
    ("Додекаэдр",      build_dodecahedron),
]
SOLID_MAP = dict(SOLIDS)


class Application:

    IMG_W = 560
    IMG_H = 560

    def __init__(self, root):
        self.root = root
        root.title("ЛР №6 - Аффинные преобразования в пространстве. Проецирование")
        root.resizable(False, False)

        self.renderer = Renderer(self.IMG_W, self.IMG_H)
        self.poly = SOLID_MAP["Гексаэдр (куб)"]()

        self._build_ui()
        self.redraw()

    # ------------------------------------------------------------------
    #  Построение интерфейса
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = tk.Frame(self.root, padx=6, pady=6)
        main.pack(fill="both", expand=True)

        # ---- слева: изображение ----
        left = tk.Frame(main)
        left.pack(side="left", fill="y")

        self.photo = tk.PhotoImage(width=self.IMG_W, height=self.IMG_H)
        self.photo.put("#101018", to=(0, 0, self.IMG_W, self.IMG_H))
        tk.Label(left, image=self.photo, bd=2, relief="sunken").pack()

        self.status = tk.Label(left, text="", anchor="w", justify="left")
        self.status.pack(fill="x", pady=(4, 0))

        # ---- справа: панель управления ----
        right = tk.Frame(main, padx=8)
        right.pack(side="right", fill="y")

        # 1. выбор многогранника
        box = tk.LabelFrame(right, text="Многогранник", padx=6, pady=4)
        box.pack(fill="x", pady=3)
        self.solid_var = tk.StringVar(value="Гексаэдр (куб)")
        cb = ttk.Combobox(box, textvariable=self.solid_var, state="readonly",
                          values=[name for name, _ in SOLIDS])
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", self.on_solid_changed)

        # 2. проекция
        box = tk.LabelFrame(right, text="Проекция", padx=6, pady=4)
        box.pack(fill="x", pady=3)
        self.proj_var = tk.StringVar(value="perspective")
        tk.Radiobutton(box, text="Перспективная", variable=self.proj_var,
                       value="perspective",
                       command=self.redraw).pack(anchor="w")
        tk.Radiobutton(box, text="Аксонометрическая (изометрия)",
                       variable=self.proj_var, value="axonometric",
                       command=self.redraw).pack(anchor="w")

        # 3. перенос
        box = tk.LabelFrame(right, text="Смещение (перенос)", padx=6, pady=4)
        box.pack(fill="x", pady=3)
        row = tk.Frame(box); row.pack(fill="x")
        self.e_dx = self._entry(row, "dx", "0.4")
        self.e_dy = self._entry(row, "dy", "0.0")
        self.e_dz = self._entry(row, "dz", "0.0")
        tk.Button(box, text="Сместить", command=self.on_translate).pack(fill="x", pady=2)

        # 4. поворот вокруг оси через центр
        box = tk.LabelFrame(right, text="Поворот вокруг оси через центр",
                            padx=6, pady=4)
        box.pack(fill="x", pady=3)
        row = tk.Frame(box); row.pack(fill="x")
        self.axis_var = tk.StringVar(value="OX")
        for ax in ("OX", "OY", "OZ"):
            tk.Radiobutton(row, text=ax, variable=self.axis_var,
                           value=ax).pack(side="left")
        row = tk.Frame(box); row.pack(fill="x")
        self.e_rot_angle = self._entry(row, "угол (°)", "25", width=8)
        tk.Button(box, text="Повернуть", command=self.on_rotate_axis).pack(fill="x", pady=2)

        # 5. масштаб
        box = tk.LabelFrame(right, text="Масштабирование", padx=6, pady=4)
        box.pack(fill="x", pady=3)
        row = tk.Frame(box); row.pack(fill="x")
        self.e_scale = self._entry(row, "k", "1.25", width=8)
        self.scale_center_var = tk.BooleanVar(value=True)
        tk.Checkbutton(box, text="относительно центра многогранника",
                       variable=self.scale_center_var).pack(anchor="w")
        tk.Button(box, text="Масштабировать", command=self.on_scale).pack(fill="x", pady=2)

        # 6. отражение
        box = tk.LabelFrame(right, text="Отражение", padx=6, pady=4)
        box.pack(fill="x", pady=3)
        row = tk.Frame(box); row.pack(fill="x")
        self.plane_var = tk.StringVar(value="OXY")
        for pl in ("OXY", "OXZ", "OYZ"):
            tk.Radiobutton(row, text=pl, variable=self.plane_var,
                           value=pl).pack(side="left")
        tk.Button(box, text="Отразить", command=self.on_reflect).pack(fill="x", pady=2)

        # 7. поворот вокруг произвольной прямой
        box = tk.LabelFrame(right, text="Поворот вокруг произвольной прямой",
                            padx=6, pady=4)
        box.pack(fill="x", pady=3)
        row = tk.Frame(box); row.pack(fill="x")
        tk.Label(row, text="P1:", width=3, anchor="w").pack(side="left")
        self.e_p1x = self._entry(row, "x", "0", width=4)
        self.e_p1y = self._entry(row, "y", "0", width=4)
        self.e_p1z = self._entry(row, "z", "0", width=4)
        row = tk.Frame(box); row.pack(fill="x")
        tk.Label(row, text="P2:", width=3, anchor="w").pack(side="left")
        self.e_p2x = self._entry(row, "x", "0", width=4)
        self.e_p2y = self._entry(row, "y", "1", width=4)
        self.e_p2z = self._entry(row, "z", "0", width=4)
        row = tk.Frame(box); row.pack(fill="x")
        self.e_line_angle = self._entry(row, "угол (°)", "30", width=8)
        tk.Button(box, text="Повернуть вокруг прямой",
                  command=self.on_rotate_line).pack(fill="x", pady=2)

        # 8. сброс
        tk.Button(right, text="Сбросить преобразования",
                  command=self.on_reset).pack(fill="x", pady=(8, 2))

    def _entry(self, parent, label, default, width=5):
        f = tk.Frame(parent)
        f.pack(side="left", padx=1)
        tk.Label(f, text=label).pack(side="left")
        e = tk.Entry(f, width=width, justify="right")
        e.insert(0, default)
        e.pack(side="left")
        return e

    # ------------------------------------------------------------------
    #  Вспомогательные функции
    # ------------------------------------------------------------------

    @staticmethod
    def _num(entry, name):
        s = entry.get().strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            raise ValueError("Поле «%s» содержит не число: %r" % (name, entry.get()))

    def _apply_matrix(self, M):
        self.poly.apply(M)
        self.redraw()

    # ------------------------------------------------------------------
    #  Обработчики кнопок
    # ------------------------------------------------------------------

    def on_solid_changed(self, _event=None):
        self.poly = SOLID_MAP[self.solid_var.get()]()
        self.redraw()

    def on_translate(self):
        try:
            dx = self._num(self.e_dx, "dx")
            dy = self._num(self.e_dy, "dy")
            dz = self._num(self.e_dz, "dz")
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        self._apply_matrix(Matrix4.translation(dx, dy, dz))

    def on_rotate_axis(self):
        try:
            ang = math.radians(self._num(self.e_rot_angle, "угол"))
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        c = self.poly.center()
        axis = self.axis_var.get()
        if axis == "OX":
            p1 = Point(c.x + 1.0, c.y, c.z)
        elif axis == "OY":
            p1 = Point(c.x, c.y + 1.0, c.z)
        else:
            p1 = Point(c.x, c.y, c.z + 1.0)
        # прямая проходит через центр многогранника параллельно оси
        self._apply_matrix(Matrix4.rotation_about_line(c, p1, ang))

    def on_scale(self):
        try:
            k = self._num(self.e_scale, "k")
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        if abs(k) < EPS:
            messagebox.showerror("Ошибка ввода", "Коэффициент масштаба не может быть нулём.")
            return
        if self.scale_center_var.get():
            c = self.poly.center()
            M = (Matrix4.translation(c.x, c.y, c.z)
                 @ Matrix4.scaling(k, k, k)
                 @ Matrix4.translation(-c.x, -c.y, -c.z))
        else:
            M = Matrix4.scaling(k, k, k)
        self._apply_matrix(M)

    def on_reflect(self):
        self._apply_matrix(Matrix4.reflection(self.plane_var.get()))

    def on_rotate_line(self):
        try:
            x1 = self._num(self.e_p1x, "P1.x")
            y1 = self._num(self.e_p1y, "P1.y")
            z1 = self._num(self.e_p1z, "P1.z")
            x2 = self._num(self.e_p2x, "P2.x")
            y2 = self._num(self.e_p2y, "P2.y")
            z2 = self._num(self.e_p2z, "P2.z")
            ang = math.radians(self._num(self.e_line_angle, "угол"))
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        p0 = Point(x1, y1, z1)
        p1 = Point(x2, y2, z2)
        if vec_len(vec_sub(p1, p0)) < EPS:
            messagebox.showerror("Ошибка ввода",
                                 "Точки P1 и P2 должны быть различны.")
            return
        self._apply_matrix(Matrix4.rotation_about_line(p0, p1, ang))

    def on_reset(self):
        self.poly.reset()
        self.redraw()

    # ------------------------------------------------------------------
    #  Перерисовка: рендер + вывод буфера в PhotoImage
    # ------------------------------------------------------------------

    def redraw(self):
        # режим проекции и согласованное с ним направление света
        self.renderer.projection.mode = self.proj_var.get()
        if self.proj_var.get() == "perspective":
            self.renderer.light = vec_norm(Point(0.35, 0.55, -1.0, 0.0))
        else:
            self.renderer.light = vec_norm(Point(-0.60, -0.80, -1.00, 0.0))

        self.renderer.render(self.poly)

        # попиксельный вывод в изображение (построчно, с кэшем цветов)
        W, H = self.IMG_W, self.IMG_H
        buf = self.renderer.buf
        cache = {}
        for y in range(H):
            base = y * W * 3
            parts = []
            for x in range(W):
                i = base + x * 3
                key = (buf[i] << 16) | (buf[i + 1] << 8) | buf[i + 2]
                s = cache.get(key)
                if s is None:
                    s = "#%06x" % key
                    cache[key] = s
                parts.append(s)
            self.photo.put("{" + " ".join(parts) + "}", to=(0, y))

        c = self.poly.center()
        self.status.config(
            text="%s | вершин: %d, граней: %d, рёбер: %d\n"
                 "центр: (%.2f, %.2f, %.2f) | проекция: %s"
                 % (self.poly.name, len(self.poly.vertices), len(self.poly.faces),
                    len(self.poly.edges()), c.x, c.y, c.z,
                    "перспективная" if self.proj_var.get() == "perspective"
                    else "аксонометрическая"))


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except tk.TclError:
        pass
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()