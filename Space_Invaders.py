# =============================================================================
#  Hailo 3D Space Invaders — Complete Version
#  Raspberry Pi 5 + Hailo AI HAT+ | Pose Estimation Multiplayer
#  Kangan Digital Initiative
# =============================================================================

import threading
import queue
import argparse
from collections import deque
import pygame
import pygame.sndarray
import random
import math
import os
import json
import numpy as np
import time
import sys

# =============================================================================
# ARGUMENT PARSING
# Normalise --input so /dev/video0 → 'usb' (hailo_apps_infra keyword)
# =============================================================================

def _parse_and_normalise_input():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', '-i', default='usb')
    args, _ = parser.parse_known_args()
    raw = args.input.strip()
    if raw.startswith('/dev/video'):
        for idx, val in enumerate(sys.argv):
            if val == raw:
                sys.argv[idx] = 'usb'
                break
        return 'usb', raw
    return raw, None

INPUT_SOURCE, CAMERA_DEVICE = _parse_and_normalise_input()

# =============================================================================
# HAILO / GSTREAMER — optional import
# Falls back to keyboard control if Hailo env is not active.
# =============================================================================

HAILO_AVAILABLE = False
_glib_loop       = None

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)

    import hailo
    from hailo_apps_infra.hailo_rpi_common import (
        get_caps_from_pad, app_callback_class
    )
    from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

    HAILO_AVAILABLE = True
    print("[INFO] Hailo AI HAT+ detected — pose estimation enabled.")
    print(f"[INFO] Input : {INPUT_SOURCE}" + (f"  ({CAMERA_DEVICE})" if CAMERA_DEVICE else ""))

except Exception as _hailo_err:
    print(f"[WARN] Hailo / hailo_apps_infra not available: {_hailo_err}")
    print("[INFO] Running in KEYBOARD fallback mode.")
    print("[INFO] To enable Hailo:  source ~/hailo-rpi5-examples/setup_env.sh")

    # Minimal stubs — keep the rest of the file parseable
    class app_callback_class:
        pass

    class _FakeApp:
        def run(self): pass

    def GStreamerPoseEstimationApp(cb, ud):
        return _FakeApp()

    class Gst:
        class PadProbeReturn:
            OK = 0

    class GLib:
        class MainLoop:
            def run(self): pass


def _start_glib_loop():
    """GStreamer needs a running GLib loop to dispatch pipeline events."""
    global _glib_loop
    if not HAILO_AVAILABLE:
        return
    _glib_loop = GLib.MainLoop()
    try:
        _glib_loop.run()
    except Exception as exc:
        print(f"[WARN] GLib loop exited: {exc}")


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60

MAX_PLAYERS      = 2
SMOOTHING_WINDOW = 8        # frames of position smoothing (lower = more responsive)
MIRROR_X         = True     # True = mirror camera so player's right → ship moves right

# --- Pose keypoints (COCO-17 format) ---
KP_NOSE           = 0
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
POSE_MIN_SCORE    = 0.4     # minimum landmark confidence to use

# --- Ship ---
SHIP_WIDTH  = 60
SHIP_HEIGHT = 40

# --- Bullets ---
BULLET_SPEED          = 12
BULLET_RADIUS         = 4
SHOOT_INTERVAL_FRAMES = 20
BULLET_TRAIL_LEN      = 6

# --- Enemies ---
ENEMY_WIDTH        = 48
ENEMY_HEIGHT       = 40
ENEMY_SPEED        = 1.5
ENEMY_SPAWN_CHANCE = 0.02
ENEMY_DRIFT_SPEED  = 1.5

# Enemy types: (speed_mult, points, ring_color, spawn_weight)
ENEMY_TYPES = [
    (1.0,  10, None,            60),   # normal
    (2.0,  20, (255,  80,  80), 25),   # fast  — red ring
    (0.5,  30, ( 80,  80, 255), 15),   # tank  — blue ring
]
_ENEMY_TYPE_WEIGHTS = [t[3] for t in ENEMY_TYPES]

# --- Round ---
ROUND_TIME = 60  # seconds

# --- Banner / Fireworks ---
FIREWORK_CHANCE      = 0.15
BANNER_SPEED         = 5
BANNER_WOBBLE_AMP    = 20
BANNER_WAVE_AMP      = 5
INITIAL_BANNER_TIMER = 120
FIREWORK_DURATION    = 20

# --- Visuals ---
SHAKE_FRAMES       = 18
SHAKE_AMPLITUDE    = 8
LOW_TIME_THRESHOLD = 10
GHOST_ALPHA        = 60

# --- Stars ---
NUM_STARS    = 180
STAR_SPEED_Z = 0.015
FOV          = 300.0

# --- Colours ---
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = (  0, 220,   0)
BLUE   = (  0, 120, 255)
YELLOW = (255, 230,   0)
PURPLE = (180,   0, 255)
ORANGE = (255, 140,   0)
RED    = (220,  30,  30)
CYAN   = (  0, 220, 220)

PLAYER_COLORS = [GREEN, BLUE, YELLOW, PURPLE]

# --- Audio ---
SAMPLE_RATE = 44100
BGM_VOLUME  = 0.35
SFX_VOLUME  = 0.70

AUDIO_FILES = {
    'shoot':     None,
    'explode_0': None,
    'explode_1': None,
    'explode_2': None,
    'round_end': None,
    'tick':      None,
    'bgm':       None,
}

HIGH_SCORE_FILE = os.path.join(os.path.dirname(__file__), '.highscore.json')


# =============================================================================
# HIGH SCORE
# =============================================================================

def load_high_score():
    try:
        with open(HIGH_SCORE_FILE, 'r') as f:
            return json.load(f).get('high_score', 0)
    except Exception:
        return 0


def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            json.dump({'high_score': score}, f)
    except Exception:
        pass


# =============================================================================
# POSE ESTIMATION INTEGRATION
# =============================================================================

class PoseInvadersUserData(app_callback_class):
    """
    Thread-safe shared state between the GStreamer callback thread
    and the main pygame game loop.

    Ship position is stored as a normalised float 0.0–1.0 (fraction of
    screen width).  The main loop scales to pixels.  This means the
    camera resolution is completely decoupled from the game resolution.
    """

    def __init__(self):
        super().__init__()
        # Normalised X positions (0.0 – 1.0) per player
        self._position_queues = [queue.Queue(maxsize=5) for _ in range(MAX_PLAYERS)]
        self._lock            = threading.Lock()
        self._active          = [False] * MAX_PLAYERS
        # Diagnostics visible in the HUD
        self.detections_last_frame = 0
        self.camera_fps            = 0.0
        self._last_frame_time      = time.time()

    # ---- position ----
    def put_position(self, player_idx, x_norm):
        """Push a normalised X position (0–1) for the given player."""
        try:
            self._position_queues[player_idx].put_nowait(float(x_norm))
        except queue.Full:
            pass

    def get_position(self, player_idx):
        """Pop the latest position; returns None if no new data."""
        try:
            return self._position_queues[player_idx].get_nowait()
        except queue.Empty:
            return None

    # ---- active flags ----
    def set_active(self, player_idx, state: bool):
        with self._lock:
            self._active[player_idx] = state

    def set_all_active(self, states):
        with self._lock:
            self._active = list(states)

    @property
    def active_players(self):
        with self._lock:
            return list(self._active)

    # ---- camera FPS bookkeeping ----
    def tick_frame(self, n_detections: int):
        now = time.time()
        dt  = now - self._last_frame_time
        if dt > 0:
            self.camera_fps = round(1.0 / dt, 1)
        self._last_frame_time      = now
        self.detections_last_frame = n_detections


class PoseInvadersCallback(app_callback_class):
    """
    GStreamer pad probe callback.

    Extracts horizontal body position from Hailo pose detections and
    pushes normalised (0–1) X coordinates into PoseInvadersUserData.

    Control point priority (most stable → least):
      1. Mid-hip  (KP_LEFT_HIP + KP_RIGHT_HIP averaged)
      2. Mid-shoulder (KP_LEFT_SHOULDER + KP_RIGHT_SHOULDER averaged)
      3. Nose     (KP_NOSE)
    """

    def __init__(self, user_data: PoseInvadersUserData):
        super().__init__()
        self.user_data = user_data

    # ------------------------------------------------------------------
    def _best_x_norm(self, pts, bbox) -> float | None:
        """
        Return the best normalised X (0–1 in camera frame, already bbox-
        expanded to full-frame coordinates) from available keypoints.
        Returns None if no reliable point found.
        """
        def kp_ok(idx):
            if idx >= len(pts):
                return False
            pt = pts[idx]
            # hailo landmark points expose a score via .score()
            try:
                return pt.score() >= POSE_MIN_SCORE
            except Exception:
                return True   # older SDK versions have no score — accept all

        def kp_x(idx):
            """Convert landmark relative-to-bbox to full-frame normalised X."""
            pt = pts[idx]
            return pt.x() * bbox.width() + bbox.xmin()

        # 1. Mid-hip
        if kp_ok(KP_LEFT_HIP) and kp_ok(KP_RIGHT_HIP):
            return (kp_x(KP_LEFT_HIP) + kp_x(KP_RIGHT_HIP)) / 2.0

        # 2. Mid-shoulder
        if kp_ok(KP_LEFT_SHOULDER) and kp_ok(KP_RIGHT_SHOULDER):
            return (kp_x(KP_LEFT_SHOULDER) + kp_x(KP_RIGHT_SHOULDER)) / 2.0

        # 3. Nose
        if kp_ok(KP_NOSE):
            return kp_x(KP_NOSE)

        return None

    # ------------------------------------------------------------------
    def __call__(self, pad, info, _u_data):
        if not HAILO_AVAILABLE:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi  = hailo.get_roi_from_buffer(buffer)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)

        positions = []   # list of (x_norm_0_to_1,)

        for det in dets:
            if det.get_label() not in ('person', ''):
                continue
            lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            pts  = lms[0].get_points()
            bbox = det.get_bbox()
            xn   = self._best_x_norm(pts, bbox)
            if xn is None:
                continue
            positions.append(xn)

        # Sort left-to-right so P1 is always the left-most person.
        # Mirror if needed (camera shows mirror image by default).
        positions.sort()
        if MIRROR_X:
            positions = [1.0 - x for x in reversed(positions)]

        # Push positions + mark active flags
        new_active = [False] * MAX_PLAYERS
        for idx, xn in enumerate(positions[:MAX_PLAYERS]):
            self.user_data.put_position(idx, xn)
            new_active[idx] = True

        self.user_data.set_all_active(new_active)
        self.user_data.tick_frame(len(positions))

        return Gst.PadProbeReturn.OK


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def _make_tone(freq, duration_ms, volume=0.5, wave='square'):
    n   = int(SAMPLE_RATE * duration_ms / 1000)
    t   = np.linspace(0, duration_ms / 1000, n, False)
    if wave == 'square':
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == 'sine':
        sig = np.sin(2 * np.pi * freq * t)
    elif wave == 'sawtooth':
        sig = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif wave == 'noise':
        sig = np.random.uniform(-1, 1, n)
    else:
        sig = np.sin(2 * np.pi * freq * t)
    fade = max(1, int(n * 0.2))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _make_sweep(f0, f1, duration_ms, volume=0.5, wave='sine'):
    n     = int(SAMPLE_RATE * duration_ms / 1000)
    freq  = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    sig   = np.sin(phase) if wave == 'sine' else np.sign(np.sin(phase))
    fade  = max(1, int(n * 0.15))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _load_or_gen(key, gen_fn):
    path = AUDIO_FILES.get(key)
    if path and os.path.isfile(path):
        snd = pygame.mixer.Sound(path)
        snd.set_volume(SFX_VOLUME)
        return snd
    return gen_fn()


def build_sounds():
    return {
        'shoot':     _load_or_gen('shoot',     lambda: _make_tone(880, 55,  SFX_VOLUME * 0.6, 'square')),
        'explode_0': _load_or_gen('explode_0', lambda: _make_tone(180, 220, SFX_VOLUME * 0.8, 'noise')),
        'explode_1': _load_or_gen('explode_1', lambda: _make_tone(320, 130, SFX_VOLUME * 0.7, 'noise')),
        'explode_2': _load_or_gen('explode_2', lambda: _make_tone(90,  380, SFX_VOLUME,       'noise')),
        'round_end': _load_or_gen('round_end', lambda: _make_sweep(220, 880, 700, SFX_VOLUME * 0.9, 'sine')),
        'tick':      _load_or_gen('tick',      lambda: _make_tone(520, 80,  SFX_VOLUME * 0.5, 'square')),
    }


def start_bgm():
    path = AUDIO_FILES.get('bgm')
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(BGM_VOLUME)
        pygame.mixer.music.play(-1)
        return
    # Synthesised drone — two detuned square waves with slow pulse
    n      = int(SAMPLE_RATE * 2.0)
    t      = np.linspace(0, 2.0, n, False)
    wave_a = np.sign(np.sin(2 * np.pi * 55.0 * t))
    wave_b = np.sign(np.sin(2 * np.pi * 55.7 * t))
    pulse  = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    sig    = ((wave_a + wave_b) * 0.5 * pulse * BGM_VOLUME * 32767).astype(np.int16)
    bgm    = pygame.sndarray.make_sound(np.column_stack([sig, sig]))
    bgm.play(loops=-1)


# =============================================================================
# STARFIELD (3D depth scroll)
# =============================================================================

def generate_stars_3d():
    cx, cy = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    return [[random.uniform(-cx, cx), random.uniform(-cy, cy), random.uniform(0.1, 1.0)]
            for _ in range(NUM_STARS)]


def draw_background_3d(surf, stars):
    surf.fill((5, 5, 18))
    cx, cy = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    for star in stars:
        star[2] -= STAR_SPEED_Z
        if star[2] <= 0:
            star[0] = random.uniform(-cx, cx)
            star[1] = random.uniform(-cy, cy)
            star[2] = 1.0
        sx = int(cx + star[0] / star[2])
        sy = int(cy + star[1] / star[2])
        if 0 <= sx < WINDOW_WIDTH and 0 <= sy < WINDOW_HEIGHT:
            brightness = int(220 * (1 - star[2]))
            size       = max(1, int(3 * (1 - star[2])))
            color      = (brightness, brightness, min(255, brightness + 40))
            if size == 1:
                surf.set_at((sx, sy), color)
            else:
                pygame.draw.circle(surf, color, (sx, sy), size)


# =============================================================================
# DRAW HELPERS
# =============================================================================

def draw_text(surf, text, x, y, color=WHITE, size=28):
    font = pygame.font.Font(None, size)
    s    = font.render(text, True, color)
    surf.blit(s, (x, y))


def draw_ship(surf, x, y, frame, color):
    cx  = x + SHIP_WIDTH  // 2
    by_ = y + SHIP_HEIGHT

    # Engine glow
    glow_size = 6 + int(4 * math.sin(frame * 0.3))
    glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*ORANGE, 120), (glow_size, glow_size), glow_size)
    surf.blit(glow_surf, (cx - glow_size, by_ - glow_size // 2))

    # Flame
    flame_h = 10 + int(5 * math.sin(frame * 0.4))
    pygame.draw.polygon(surf, ORANGE, [(cx, by_ + flame_h), (cx - 6, by_), (cx + 6, by_)])
    pygame.draw.polygon(surf, YELLOW, [(cx, by_ + flame_h - 3), (cx - 3, by_), (cx + 3, by_)])

    # Hull
    tip, bl, br = (cx, y), (x, by_), (x + SHIP_WIDTH, by_)
    pygame.draw.polygon(surf, color, [tip, bl, br])
    pygame.draw.polygon(surf, WHITE,  [tip, bl, br], 2)

    # Cockpit
    dome_r = SHIP_WIDTH // 6
    pygame.draw.circle(surf, WHITE, (cx, y + SHIP_HEIGHT // 2), dome_r)
    pygame.draw.circle(surf, color, (cx, y + SHIP_HEIGHT // 2), dome_r - 2)

    # Wing accents
    wing_col = tuple(min(255, c + 60) for c in color)
    pygame.draw.line(surf, wing_col, (cx, y + SHIP_HEIGHT // 2), bl, 2)
    pygame.draw.line(surf, wing_col, (cx, y + SHIP_HEIGHT // 2), br, 2)


def draw_enemy(surf, x, y):
    cx = int(x + ENEMY_WIDTH  // 2)
    cy = int(y + ENEMY_HEIGHT // 2)
    body_pts = [
        (cx - 10, y), (cx + 10, y),
        (cx + ENEMY_WIDTH // 2, cy),
        (cx + 10, y + ENEMY_HEIGHT), (cx - 10, y + ENEMY_HEIGHT),
        (cx - ENEMY_WIDTH // 2, cy),
    ]
    pygame.draw.polygon(surf, (0, 200, 80), body_pts)
    pygame.draw.polygon(surf, (0, 255, 120), body_pts, 2)
    pygame.draw.circle(surf, RED, (cx - 8, cy - 4), 5)
    pygame.draw.circle(surf, RED, (cx + 8, cy - 4), 5)
    pygame.draw.circle(surf, (255, 100, 100), (cx - 8, cy - 4), 2)
    pygame.draw.circle(surf, (255, 100, 100), (cx + 8, cy - 4), 2)
    pygame.draw.line(surf, (0, 200, 80), (cx - 8, y),  (cx - 14, y - 10), 2)
    pygame.draw.line(surf, (0, 200, 80), (cx + 8, y),  (cx + 14, y - 10), 2)
    pygame.draw.circle(surf, YELLOW, (cx - 14, y - 10), 3)
    pygame.draw.circle(surf, YELLOW, (cx + 14, y - 10), 3)
    for k in range(3):
        lx = cx - 8 + k * 8
        pygame.draw.line(surf, (0, 150, 50), (lx, cy + 4), (lx, cy + 10), 2)


def draw_explosion(surf, cx, cy, t):
    num_particles = 14
    for k in range(num_particles):
        angle  = 2 * math.pi * k / num_particles + t * 0.1
        dist   = (20 - t) * 1.8 + random.uniform(0, 6)
        px     = int(cx + math.cos(angle) * dist)
        py     = int(cy + math.sin(angle) * dist)
        radius = max(1, t // 3)
        heat   = min(255, t * 18)
        color  = (heat, max(0, heat - 80), 0)
        if 0 <= px < WINDOW_WIDTH and 0 <= py < WINDOW_HEIGHT:
            pygame.draw.circle(surf, color, (px, py), radius)
    if t > 10:
        core_r = (t - 10) * 2
        flash  = pygame.Surface((core_r * 2, core_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(flash, (255, 255, 200, min(200, t * 15)), (core_r, core_r), core_r)
        surf.blit(flash, (int(cx) - core_r, int(cy) - core_r))


def _draw_bullet_trail(surf, bullet, player_color):
    trail = bullet.get('trail', [])
    n     = len(trail)
    if n < 2:
        return
    for k in range(n - 1):
        ratio = k / n
        r     = min(255, int(player_color[0] * ratio + 255 * (1 - ratio)))
        g     = min(255, int(player_color[1] * ratio))
        b     = min(255, int(player_color[2] * ratio))
        w_px  = max(1, int(BULLET_RADIUS * ratio))
        pygame.draw.line(surf, (r, g, b),
                         (int(trail[k][0]),   int(trail[k][1])),
                         (int(trail[k+1][0]), int(trail[k+1][1])), w_px)


def _draw_enemy_overlay(surf, en):
    _, _, ring_color, _ = ENEMY_TYPES[en['type']]
    if ring_color is None:
        return
    cx     = int(en['x'] + ENEMY_WIDTH  // 2)
    cy     = int(en['y'] + ENEMY_HEIGHT // 2)
    radius = ENEMY_WIDTH // 2 + 4
    pulse  = int(3 * math.sin(en['y'] * 0.05 + en['phase']))
    pygame.draw.circle(surf, ring_color, (cx, cy), radius + pulse, 2)


def _draw_score_popups(surf, popups, popup_font):
    for p in popups:
        alpha     = int(255 * p['t'] / p['max_t'])
        col       = tuple(min(255, int(c * alpha / 255)) for c in p['color'])
        text_surf = popup_font.render(p['text'], True, col)
        surf.blit(text_surf, (int(p['x']) - text_surf.get_width() // 2, int(p['y'])))


def _draw_ghost_ship(surf, x, color):
    ghost = pygame.Surface((SHIP_WIDTH, SHIP_HEIGHT), pygame.SRCALPHA)
    cx    = SHIP_WIDTH // 2
    pts   = [(cx, 0), (0, SHIP_HEIGHT), (SHIP_WIDTH, SHIP_HEIGHT)]
    pygame.draw.polygon(ghost, (*color, GHOST_ALPHA),      pts)
    pygame.draw.polygon(ghost, (*color, GHOST_ALPHA + 40), pts, 2)
    surf.blit(ghost, (x, WINDOW_HEIGHT - SHIP_HEIGHT))


def _draw_hud(surf, scores, high_score, t_left, frame, active_players,
              user_data=None, hailo_ok=False):
    # Player scores
    for i, score in enumerate(scores):
        label = f"P{i+1}: {score}" + ("" if active_players[i] else " (away)")
        draw_text(surf, label, 10, 10 + i * 32, PLAYER_COLORS[i])

    # High score
    draw_text(surf, f"BEST: {high_score}", 10, 10 + MAX_PLAYERS * 32 + 8, WHITE)

    # Timer
    if t_left <= LOW_TIME_THRESHOLD and frame % 20 < 10:
        timer_col = (255, 50, 50)
    else:
        timer_col = WHITE
    draw_text(surf, f"TIME: {t_left:02d}", WINDOW_WIDTH - 160, 10, timer_col)

    # Active player dots
    for i in range(MAX_PLAYERS):
        dot_col = PLAYER_COLORS[i] if active_players[i] else (60, 60, 60)
        pygame.draw.circle(surf, dot_col, (WINDOW_WIDTH - 20, 20 + i * 18), 6)

    # Camera diagnostics (bottom-left, only when Hailo is running)
    if hailo_ok and user_data is not None:
        cam_label = (
            f"CAM  {user_data.camera_fps:.0f} fps  |  "
            f"{user_data.detections_last_frame} person(s) detected  |  "
            f"src: {INPUT_SOURCE}"
        )
        draw_text(surf, cam_label, 8, WINDOW_HEIGHT - 24, (150, 150, 180), 20)


def _draw_pose_debug(surf, active_players, ship_positions):
    """
    Draw a thin position bar at the bottom of the screen so players
    can see their detected position even before a ship is active.
    """
    bar_y  = WINDOW_HEIGHT - SHIP_HEIGHT - 12
    bar_h  = 4
    for i in range(MAX_PLAYERS):
        col     = PLAYER_COLORS[i]
        ship_cx = ship_positions[i] + SHIP_WIDTH // 2
        # Draw a small triangle marker above the bar
        pts = [(ship_cx, bar_y - 6), (ship_cx - 5, bar_y), (ship_cx + 5, bar_y)]
        alpha_col = tuple(max(0, c - 80) for c in col) if not active_players[i] else col
        pygame.draw.polygon(surf, alpha_col, pts)


# =============================================================================
# MAIN GAME LOOP
# =============================================================================

def main():
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    screen    = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hailo 3D Space Invaders — Kangan Digital Initiative")
    clock     = pygame.time.Clock()
    stars3d   = generate_stars_3d()
    game_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    banner_font = pygame.font.Font(None, 72)
    popup_font  = pygame.font.Font(None, 36)

    sounds = build_sounds()
    start_bgm()

    bullet_outer_colors = [tuple(min(255, c + 80) for c in col) for col in PLAYER_COLORS]

    # Game state
    # ship_positions stored as pixel X of left edge of ship
    default_xs         = [WINDOW_WIDTH // (MAX_PLAYERS + 1) * (i + 1)
                          for i in range(MAX_PLAYERS)]
    # position_histories stores normalised 0–1 values
    position_histories = [
        deque([default_xs[i] / WINDOW_WIDTH] * SMOOTHING_WINDOW,
              maxlen=SMOOTHING_WINDOW)
        for i in range(MAX_PLAYERS)
    ]
    ship_positions     = list(default_xs)
    bullets            = [[] for _ in range(MAX_PLAYERS)]
    enemies            = []
    explosions         = []
    fireworks          = []
    score_popups       = []
    scores             = [0] * MAX_PLAYERS
    high_score         = load_high_score()
    shoot_cooldowns    = [0] * MAX_PLAYERS

    start_time       = time.time()
    frame            = 0
    banner_timer     = 0
    banner_y         = -100
    shake_timer      = 0
    last_tick_second = -1

    # ------------------------------------------------------------------
    # CAMERA + POSE ESTIMATION STARTUP
    # ------------------------------------------------------------------
    user_data = PoseInvadersUserData()
    _hailo_running = False

    if HAILO_AVAILABLE:
        if CAMERA_DEVICE and not os.path.exists(CAMERA_DEVICE):
            print(f"[ERROR] Camera device not found: {CAMERA_DEVICE}")
            print( "        Run: ls /dev/video*")
            print( "[INFO]  Falling back to keyboard control.")
        else:
            # 1. GLib main loop must be running before the pipeline starts
            glib_thread = threading.Thread(
                target=_start_glib_loop, daemon=True, name='GLib-Loop'
            )
            glib_thread.start()
            time.sleep(0.15)   # give GLib time to spin up

            # 2. Build and start the pose pipeline in its own thread
            def _run_pose_app():
                try:
                    cb  = PoseInvadersCallback(user_data)
                    app = GStreamerPoseEstimationApp(cb, user_data)
                    app.run()
                except Exception as exc:
                    print(f"[ERROR] Pose pipeline crashed: {exc}")
                    print( "[INFO]  Falling back to keyboard control.")

            pose_thread = threading.Thread(
                target=_run_pose_app, daemon=True, name='Pose-Pipeline'
            )
            pose_thread.start()
            _hailo_running = True
            print("[INFO] Camera pipeline starting…  (first frame may take 2–3 s)")
            print("[INFO] Control: move your BODY left/right in front of the camera.")
    else:
        print("[INFO] Keyboard fallback active.")
        print("       P1: A / D    P2: LEFT / RIGHT")

    BGM_RESTART_EVENT = pygame.USEREVENT + 1

    running = True
    while running:
        # ----------------------------------------------------------------
        # EVENTS
        # ----------------------------------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == BGM_RESTART_EVENT:
                start_bgm()

        # ----------------------------------------------------------------
        # KEYBOARD FALLBACK  (always checked — allows hybrid control)
        # ----------------------------------------------------------------
        keys = pygame.key.get_pressed()
        kb   = [(pygame.K_a, pygame.K_d), (pygame.K_LEFT, pygame.K_RIGHT)]
        for i, (kl, kr) in enumerate(kb):
            if i >= MAX_PLAYERS:
                break
            moved = False
            if keys[kl]:
                new_xn = max(0.0, position_histories[i][-1] - 8.0 / WINDOW_WIDTH)
                position_histories[i].append(new_xn)
                moved = True
            elif keys[kr]:
                new_xn = min(1.0, position_histories[i][-1] + 8.0 / WINDOW_WIDTH)
                position_histories[i].append(new_xn)
                moved = True
            # In keyboard-only mode mark player as active once a key is pressed
            if moved and not _hailo_running:
                user_data.set_active(i, True)

        # ----------------------------------------------------------------
        # READ POSE POSITIONS FROM HAILO
        # Each player's position arrives as a normalised 0–1 float.
        # ----------------------------------------------------------------
        if _hailo_running:
            for i in range(MAX_PLAYERS):
                xn = user_data.get_position(i)
                if xn is not None:
                    position_histories[i].append(xn)

        # Compute smoothed pixel positions
        ship_positions = [
            int(np.clip(
                (sum(hist) / len(hist)) * WINDOW_WIDTH - SHIP_WIDTH // 2,
                0, WINDOW_WIDTH - SHIP_WIDTH
            ))
            for hist in position_histories
        ]

        # ----------------------------------------------------------------
        # ACTIVE PLAYER FLAGS
        # ----------------------------------------------------------------
        active_players = (
            user_data.active_players if _hailo_running
            else [True] * MAX_PLAYERS
        )

        # ----------------------------------------------------------------
        # TIMING
        # ----------------------------------------------------------------
        elapsed = time.time() - start_time
        t_left  = max(0, int(ROUND_TIME - elapsed))

        # ----------------------------------------------------------------
        # ROUND END
        # ----------------------------------------------------------------
        if t_left == 0:
            explosions.append({'x': WINDOW_WIDTH / 2, 'y': WINDOW_HEIGHT / 2, 't': 30})
            shake_timer = SHAKE_FRAMES
            sounds['round_end'].play()
            pygame.mixer.stop()
            start_bgm()
            if max(scores) > high_score:
                high_score = max(scores)
                save_high_score(high_score)
            scores         = [0] * MAX_PLAYERS
            enemies.clear()
            [bl.clear() for bl in bullets]
            score_popups.clear()
            start_time       = time.time()
            banner_timer     = INITIAL_BANNER_TIMER
            banner_y         = -100
            last_tick_second = -1

        # ----------------------------------------------------------------
        # LOW-TIME TICK  (once per second)
        # ----------------------------------------------------------------
        if 0 < t_left <= LOW_TIME_THRESHOLD and t_left != last_tick_second:
            sounds['tick'].play()
            last_tick_second = t_left

        # ----------------------------------------------------------------
        # SHOOTING
        # ----------------------------------------------------------------
        for i in range(MAX_PLAYERS):
            if active_players[i]:
                shoot_cooldowns[i] -= 1
                if shoot_cooldowns[i] <= 0:
                    lx = ship_positions[i] + SHIP_WIDTH * 0.2
                    rx = ship_positions[i] + SHIP_WIDTH * 0.8
                    yb = WINDOW_HEIGHT - SHIP_HEIGHT * 0.5
                    bullets[i].append({'x': lx, 'y': yb, 'trail': [], 'color': PLAYER_COLORS[i]})
                    bullets[i].append({'x': rx, 'y': yb, 'trail': [], 'color': PLAYER_COLORS[i]})
                    shoot_cooldowns[i] = SHOOT_INTERVAL_FRAMES
                    sounds['shoot'].play()

        # ----------------------------------------------------------------
        # MOVE BULLETS
        # ----------------------------------------------------------------
        for blist in bullets:
            for b in blist[:]:
                b['trail'].append((b['x'], b['y']))
                if len(b['trail']) > BULLET_TRAIL_LEN:
                    b['trail'].pop(0)
                b['y'] -= BULLET_SPEED
                if b['y'] < 0:
                    blist.remove(b)

        # ----------------------------------------------------------------
        # SPAWN ENEMIES
        # ----------------------------------------------------------------
        if random.random() < ENEMY_SPAWN_CHANCE:
            etype = random.choices(range(len(ENEMY_TYPES)), weights=_ENEMY_TYPE_WEIGHTS)[0]
            enemies.append({
                'x':     random.randint(0, WINDOW_WIDTH - ENEMY_WIDTH),
                'y':     0.0,
                'type':  etype,
                'phase': random.uniform(0, math.pi * 2),
            })

        # ----------------------------------------------------------------
        # MOVE ENEMIES + COLLISION DETECTION
        # ----------------------------------------------------------------
        for en in enemies[:]:
            speed_mult, points, _, _ = ENEMY_TYPES[en['type']]
            en['y'] += ENEMY_SPEED * speed_mult
            drift    = math.sin(en['y'] * 0.02 + en['phase']) * ENEMY_DRIFT_SPEED
            en['x']  = float(np.clip(en['x'] + drift, 0, WINDOW_WIDTH - ENEMY_WIDTH))

            if en['y'] > WINDOW_HEIGHT:
                enemies.remove(en)
                continue

            ecx = int(en['x'] + ENEMY_WIDTH  // 2)
            ecy = int(en['y'] + ENEMY_HEIGHT // 2)
            hit = False
            for i, blist in enumerate(bullets):
                for b in blist[:]:
                    if math.hypot(b['x'] - ecx, b['y'] - ecy) < ENEMY_WIDTH / 2:
                        blist.remove(b)
                        enemies.remove(en)
                        explosions.append({'x': ecx, 'y': ecy, 't': 15})
                        scores[i] += points
                        score_popups.append({
                            'x': ecx, 'y': ecy,
                            'text': f'+{points}',
                            'color': PLAYER_COLORS[i],
                            't': 40, 'max_t': 40,
                        })
                        sounds[f'explode_{en["type"]}'].play()
                        hit = True
                        break
                if hit:
                    break

        # ----------------------------------------------------------------
        # AGE EXPLOSIONS & POPUPS
        # ----------------------------------------------------------------
        for ex in explosions[:]:
            ex['t'] -= 1
            if ex['t'] <= 0:
                explosions.remove(ex)

        for p in score_popups[:]:
            p['y'] -= 1.2
            p['t'] -= 1
            if p['t'] <= 0:
                score_popups.remove(p)

        if shake_timer > 0:
            shake_timer -= 1

        # ================================================================
        # DRAW
        # ================================================================
        draw_background_3d(game_surf, stars3d)

        # Ghost ships for inactive players
        for i in range(MAX_PLAYERS):
            if not active_players[i]:
                _draw_ghost_ship(game_surf, ship_positions[i], PLAYER_COLORS[i])

        # Active ships + bullets
        for i in range(MAX_PLAYERS):
            if active_players[i]:
                sy = WINDOW_HEIGHT - SHIP_HEIGHT
                draw_ship(game_surf, ship_positions[i], sy, frame, PLAYER_COLORS[i])
                for b in bullets[i]:
                    _draw_bullet_trail(game_surf, b, PLAYER_COLORS[i])
                    bx, by_ = int(b['x']), int(b['y'])
                    fh = BULLET_RADIUS * 6
                    br = BULLET_RADIUS
                    pygame.draw.polygon(game_surf, bullet_outer_colors[i],
                                        [(bx, by_), (bx - br, by_ + fh), (bx + br, by_ + fh)])
                    pygame.draw.polygon(game_surf, PLAYER_COLORS[i],
                                        [(bx, by_ + int(fh * 0.4)),
                                         (bx - br // 2, by_ + fh),
                                         (bx + br // 2, by_ + fh)])

        # Enemies
        for en in enemies:
            draw_enemy(game_surf, en['x'], en['y'])
            _draw_enemy_overlay(game_surf, en)

        # Explosions
        for ex in explosions:
            draw_explosion(game_surf, ex['x'], ex['y'], ex['t'])

        # Score popups
        _draw_score_popups(game_surf, score_popups, popup_font)

        # Pose position debug markers (always visible so players can
        # see their tracked position even before they're fully active)
        if _hailo_running:
            _draw_pose_debug(game_surf, active_players, ship_positions)

        # End-of-round banner
        if banner_timer > 0:
            banner_y = min(200, banner_y + BANNER_SPEED)
            progress = INITIAL_BANNER_TIMER - banner_timer
            wobble   = math.sin(progress / 10) * BANNER_WOBBLE_AMP
            text     = "KANGAN DIGITAL INITIATIVE"
            total_w  = banner_font.size(text)[0]
            start_x  = WINDOW_WIDTH / 2 - total_w / 2 + wobble
            for j, ch in enumerate(text):
                ch_surf = banner_font.render(ch, True, YELLOW)
                ch_x    = start_x + banner_font.size(text[:j])[0]
                ch_y    = banner_y + 30 + math.sin((progress + j * 5) / 5) * BANNER_WAVE_AMP
                game_surf.blit(ch_surf, (ch_x, ch_y))
            if random.random() < FIREWORK_CHANCE:
                fireworks.append({
                    'x': random.uniform(0, WINDOW_WIDTH),
                    'y': banner_y + random.uniform(10, 90),
                    't': FIREWORK_DURATION,
                })
            for fw in fireworks[:]:
                draw_explosion(game_surf, fw['x'], fw['y'], fw['t'])
                fw['t'] -= 1
                if fw['t'] <= 0:
                    fireworks.remove(fw)
            banner_timer -= 1

        # HUD
        _draw_hud(game_surf, scores, high_score, t_left, frame,
                  active_players, user_data=user_data, hailo_ok=_hailo_running)

        # Screen shake
        sx, sy = 0, 0
        if shake_timer > 0:
            amp = int(SHAKE_AMPLITUDE * shake_timer / SHAKE_FRAMES)
            sx  = random.randint(-amp, amp)
            sy  = random.randint(-amp, amp)

        screen.fill(BLACK)
        screen.blit(game_surf, (sx, sy))

        # Camera warm-up overlay (first 3 s)
        if _hailo_running and frame < FPS * 3:
            _font_sm  = pygame.font.SysFont('monospace', 18, bold=True)
            _cam_text = _font_sm.render(
                f"Camera warming up… ({INPUT_SOURCE})  —  step in front of the camera",
                True, (255, 220, 60)
            )
            screen.blit(_cam_text, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    # ----------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------
    save_high_score(high_score)
    pygame.mixer.stop()
    pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

