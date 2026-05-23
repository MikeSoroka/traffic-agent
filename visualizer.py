"""
Pygame visualiser for the 4-way intersection.

Call Visualizer.draw(env, ...) each simulation step.
"""
import pygame
from config import (
    SCREEN_W, SCREEN_H, FPS, COLORS,
    CX, CY, ROAD_HALF, PANEL_X,
)

# ── Derived geometry ──────────────────────────────────────────────────────────
_RX0 = CX - ROAD_HALF   # intersection box left   (330)
_RX1 = CX + ROAD_HALF   # intersection box right  (470)
_RY0 = CY - ROAD_HALF   # intersection box top    (280)
_RY1 = CY + ROAD_HALF   # intersection box bottom (420)

# Inbound lane centre lines (right-hand traffic convention)
_LANE_X_N = CX + 35    # southbound (approaching from N) — east half of N road
_LANE_X_S = CX - 35    # northbound (approaching from S) — west half
_LANE_Y_E = CY - 35    # westbound  (approaching from E) — north half of E road
_LANE_Y_W = CY + 35    # eastbound  (approaching from W) — south half

# Car dimensions
_CW_NS = 20   # width of a car in N/S lane
_CH_NS = 14   # height (along road axis)
_CW_EW = 14
_CH_EW = 20

_MAX_VIS = 7            # max cars drawn per direction
_CAR_GAP = 4            # gap between cars (px)

# Traffic-light circle positions (at stop line of each approach)
_LIGHT_N = (_LANE_X_N, _RY0 - 18)
_LIGHT_S = (_LANE_X_S, _RY1 + 18)
_LIGHT_E = (_RX1 + 18, _LANE_Y_E)
_LIGHT_W = (_RX0 - 18, _LANE_Y_W)
_LIGHT_R = 12


def make_screen() -> tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Adaptive Traffic Signal Control — Q-Learning")
    return screen, pygame.time.Clock()


class Visualizer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self._font_md = pygame.font.SysFont("monospace", 15)
        self._font_sm = pygame.font.SysFont("monospace", 12)
        self._reward_history: list[float] = []

    # ── Main draw call ────────────────────────────────────────────────────────

    def draw(self, env, episode: int = 0, step: int = 0,
             total_r: float = 0.0, label: str = "Q-Learning"):
        self._reward_history.append(total_r)
        s = self.screen
        s.fill(COLORS["bg"])
        self._draw_roads(s)
        self._draw_markings(s)
        self._draw_cars(s, env)
        self._draw_lights(s, env)
        self._draw_panel(s, env, episode, step, total_r, label)
        pygame.display.flip()

    # ── Road geometry ─────────────────────────────────────────────────────────

    def _draw_roads(self, s: pygame.Surface):
        c_road  = COLORS["road"]
        c_inter = COLORS["inter"]
        # N road
        pygame.draw.rect(s, c_road, (_RX0, 0, ROAD_HALF * 2, _RY0))
        # S road
        pygame.draw.rect(s, c_road, (_RX0, _RY1, ROAD_HALF * 2, SCREEN_H - _RY1))
        # E road
        pygame.draw.rect(s, c_road, (_RX1, _RY0, PANEL_X - _RX1, ROAD_HALF * 2))
        # W road
        pygame.draw.rect(s, c_road, (0, _RY0, _RX0, ROAD_HALF * 2))
        # Intersection box (slightly lighter)
        pygame.draw.rect(s, c_inter, (_RX0, _RY0, ROAD_HALF * 2, ROAD_HALF * 2))

    def _draw_markings(self, s: pygame.Surface):
        mc = COLORS["marking"]
        dash, gap = 14, 10
        # Centre line N road (dashed vertical)
        y = 0
        while y < _RY0:
            if y + dash <= _RY0:
                pygame.draw.line(s, mc, (CX, y), (CX, min(y + dash, _RY0)), 1)
            y += dash + gap
        # Centre line S road
        y = _RY1
        while y < SCREEN_H:
            pygame.draw.line(s, mc, (CX, y), (min(y + dash, SCREEN_H), y), 1)
            # vertical
            pygame.draw.line(s, mc, (CX, y), (CX, min(y + dash, SCREEN_H)), 1)
            y += dash + gap
        # Centre line E road (dashed horizontal)
        x = _RX1
        while x < PANEL_X:
            pygame.draw.line(s, mc, (x, CY), (min(x + dash, PANEL_X), CY), 1)
            x += dash + gap
        # Centre line W road
        x = 0
        while x < _RX0:
            pygame.draw.line(s, mc, (x, CY), (min(x + dash, _RX0), CY), 1)
            x += dash + gap

    # ── Cars ──────────────────────────────────────────────────────────────────

    def _draw_cars(self, s: pygame.Surface, env):
        q = env.q
        c_ns = COLORS["car_ns"]
        c_ew = COLORS["car_ew"]

        # N queue — cars stack upward from stop line
        for i in range(min(q["N"], _MAX_VIS)):
            y = _RY0 - 10 - i * (_CH_NS + _CAR_GAP) - _CH_NS
            r = pygame.Rect(_LANE_X_N - _CW_NS // 2, y, _CW_NS, _CH_NS)
            pygame.draw.rect(s, c_ns, r, border_radius=3)
            # windscreen highlight
            pygame.draw.rect(s, _lighter(c_ns), (r.x + 3, r.y + 2, r.w - 6, 4), border_radius=2)

        # S queue — stack downward
        for i in range(min(q["S"], _MAX_VIS)):
            y = _RY1 + 10 + i * (_CH_NS + _CAR_GAP)
            r = pygame.Rect(_LANE_X_S - _CW_NS // 2, y, _CW_NS, _CH_NS)
            pygame.draw.rect(s, c_ns, r, border_radius=3)
            pygame.draw.rect(s, _lighter(c_ns), (r.x + 3, r.bottom - 6, r.w - 6, 4), border_radius=2)

        # E queue — stack rightward
        for i in range(min(q["E"], _MAX_VIS)):
            x = _RX1 + 10 + i * (_CW_EW + _CAR_GAP)
            r = pygame.Rect(x, _LANE_Y_E - _CH_EW // 2, _CW_EW, _CH_EW)
            pygame.draw.rect(s, c_ew, r, border_radius=3)
            pygame.draw.rect(s, _lighter(c_ew), (r.x + 2, r.y + 3, 4, r.h - 6), border_radius=2)

        # W queue — stack leftward
        for i in range(min(q["W"], _MAX_VIS)):
            x = _RX0 - 10 - i * (_CW_EW + _CAR_GAP) - _CW_EW
            r = pygame.Rect(x, _LANE_Y_W - _CH_EW // 2, _CW_EW, _CH_EW)
            pygame.draw.rect(s, c_ew, r, border_radius=3)
            pygame.draw.rect(s, _lighter(c_ew), (r.right - 6, r.y + 3, 4, r.h - 6), border_radius=2)

    # ── Traffic lights ────────────────────────────────────────────────────────

    def _draw_lights(self, s: pygame.Surface, env):
        ns_c = COLORS[env.ns_light]
        ew_c = COLORS[env.ew_light]

        for pos in (_LIGHT_N, _LIGHT_S):
            pygame.draw.circle(s, (30, 30, 40), pos, _LIGHT_R + 3)  # housing
            pygame.draw.circle(s, ns_c, pos, _LIGHT_R)
            if env.ns_light == "green":
                pygame.draw.circle(s, _lighter(ns_c), (pos[0] - 3, pos[1] - 3), _LIGHT_R // 3)

        for pos in (_LIGHT_E, _LIGHT_W):
            pygame.draw.circle(s, (30, 30, 40), pos, _LIGHT_R + 3)
            pygame.draw.circle(s, ew_c, pos, _LIGHT_R)
            if env.ew_light == "green":
                pygame.draw.circle(s, _lighter(ew_c), (pos[0] - 3, pos[1] - 3), _LIGHT_R // 3)

        # Direction labels near lights
        font = self._font_sm
        for lbl, pos in [("N", _LIGHT_N), ("S", _LIGHT_S)]:
            t = font.render(lbl, True, COLORS["text_dim"])
            s.blit(t, (pos[0] - t.get_width() // 2, pos[1] - _LIGHT_R - 14))
        for lbl, pos in [("E", _LIGHT_E), ("W", _LIGHT_W)]:
            t = font.render(lbl, True, COLORS["text_dim"])
            s.blit(t, (pos[0] - t.get_width() // 2, pos[1] - _LIGHT_R - 14))

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _draw_panel(self, s: pygame.Surface, env, episode, step, total_r, label):
        pw = SCREEN_W - PANEL_X
        ph = SCREEN_H
        pygame.draw.rect(s, COLORS["panel_bg"], (PANEL_X, 0, pw, ph))
        pygame.draw.line(s, COLORS["road"], (PANEL_X, 0), (PANEL_X, ph), 2)

        x0 = PANEL_X + 12
        y  = 16

        def text(txt, font=None, color="text", dy=22):
            nonlocal y
            f = font or self._font_md
            surf = f.render(txt, True, COLORS[color])
            s.blit(surf, (x0, y))
            y += dy

        # Title
        text("TRAFFIC CTRL", self._font_lg, dy=28)
        text(label, self._font_sm, "text_dim", dy=24)

        pygame.draw.line(s, COLORS["road"], (x0, y), (SCREEN_W - 12, y), 1)
        y += 10

        # Phase indicator
        phase_color = "green" if "GREEN" in env.phase_label else "yellow"
        text(env.phase_label, color=phase_color, dy=24)

        pygame.draw.line(s, COLORS["road"], (x0, y), (SCREEN_W - 12, y), 1)
        y += 10

        # Queue bars
        text("QUEUES", self._font_sm, "text_dim", dy=20)
        self._draw_queue_bar(s, x0, y, "N", env.q["N"], COLORS["car_ns"])
        y += 22
        self._draw_queue_bar(s, x0, y, "S", env.q["S"], COLORS["car_ns"])
        y += 22
        self._draw_queue_bar(s, x0, y, "E", env.q["E"], COLORS["car_ew"])
        y += 22
        self._draw_queue_bar(s, x0, y, "W", env.q["W"], COLORS["car_ew"])
        y += 28

        pygame.draw.line(s, COLORS["road"], (x0, y), (SCREEN_W - 12, y), 1)
        y += 10

        total_q = sum(env.q.values())
        text(f"Total queue : {total_q:2d}", dy=22)
        text(f"Episode     : {episode}", dy=22)
        text(f"Step        : {step}", dy=22)
        text(f"Cum. reward : {total_r:.0f}", dy=22)

        # Mini reward sparkline
        if len(self._reward_history) > 4:
            y += 8
            self._draw_sparkline(s, x0, y, pw - 24)
            y += 60

    def _draw_queue_bar(self, s, x, y, label, count, color):
        from config import MAX_QUEUE
        bar_w = SCREEN_W - PANEL_X - 24 - 22
        filled = int(bar_w * min(count, MAX_QUEUE) / MAX_QUEUE)
        pygame.draw.rect(s, COLORS["bar_bg"], (x + 18, y + 4, bar_w, 14), border_radius=3)
        if filled > 0:
            pygame.draw.rect(s, color, (x + 18, y + 4, filled, 14), border_radius=3)
        lbl = self._font_sm.render(f"{label} {count:2d}", True, COLORS["text"])
        s.blit(lbl, (x, y + 2))

    def _draw_sparkline(self, s, x, y, width):
        h = 50
        pygame.draw.rect(s, COLORS["bar_bg"], (x, y, width, h), border_radius=4)
        data = self._reward_history[-width:]
        if len(data) < 2:
            return
        mn, mx = min(data), max(data)
        rng = mx - mn or 1
        pts = []
        for i, v in enumerate(data):
            px = x + int(i * width / len(data))
            py = y + h - int((v - mn) / rng * (h - 6)) - 3
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(s, COLORS["car_ns"], False, pts, 1)
        lbl = self._font_sm.render("reward", True, COLORS["text_dim"])
        s.blit(lbl, (x + 2, y + 2))


# ── Utility ───────────────────────────────────────────────────────────────────

def _lighter(color: tuple, amount: int = 60) -> tuple:
    return tuple(min(255, c + amount) for c in color)
