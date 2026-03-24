import threading
import queue
from collections import deque
import pygame
import pygame.sndarray
import random
import math
import gi
import os
import numpy as np
import time

import hailo
from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer, app_callback_class
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# === (Existing code unchanged) ===
# --- [keep all your previous code up to draw_text()] ---

# === Multiplayer Hailo Space Invaders Main Code ===

# --- Spawn / gameplay tuning ---
ENEMY_SPAWN_CHANCE   = 0.02   # probability per frame a new enemy spawns
FIREWORK_CHANCE      = 0.15   # probability per frame a firework is added during banner
BANNER_SPEED         = 5      # pixels per frame the banner slides down
BANNER_WOBBLE_AMP    = 20     # horizontal wobble amplitude (pixels)
BANNER_WAVE_AMP      = 5      # vertical wave amplitude per character (pixels)
INITIAL_BANNER_TIMER = 120    # frames the end-of-round banner is shown
FIREWORK_DURATION    = 20     # lifetime (frames) of each firework explosion

# --- Enemy types: (speed_multiplier, points, ring_color, spawn_weight) ---
# ring_color=None means no ring overlay (normal enemy)
ENEMY_TYPES = [
    (1.0,  10, None,              60),  # normal
    (2.0,  20, (255,  80,  80),   25),  # fast  — red ring
    (0.5,  30, ( 80,  80, 255),   15),  # tank  — blue ring
]
_ENEMY_TYPE_WEIGHTS = [t[3] for t in ENEMY_TYPES]

# --- Visual tweaks ---
BULLET_TRAIL_LEN   = 6    # number of trail segments behind each bullet
SHAKE_FRAMES       = 18   # duration of screen shake (frames)
SHAKE_AMPLITUDE    = 8    # max pixel offset during shake
LOW_TIME_THRESHOLD = 10   # seconds left before timer starts flashing
ENEMY_DRIFT_SPEED  = 1.5  # pixels per frame of sinusoidal horizontal drift
GHOST_ALPHA        = 60   # opacity of inactive-player ghost ship indicator

# --- Audio ---
SAMPLE_RATE    = 44100
BGM_VOLUME     = 0.35   # background music volume (0.0 – 1.0)
SFX_VOLUME     = 0.7    # sound effects volume (0.0 – 1.0)
# To use real audio files instead of generated tones, place them in the same
# directory as this script and set the paths below (set to None to keep tones):
AUDIO_FILES = {
    'shoot':      None,   # e.g. 'sounds/shoot.wav'
    'explode_0':  None,   # normal enemy explosion
    'explode_1':  None,   # fast enemy explosion
    'explode_2':  None,   # tank enemy explosion
    'round_end':  None,   # end-of-round fanfare
    'tick':       None,   # low-time countdown tick
    'bgm':        None,   # background music (ogg recommended)
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _make_tone(freq, duration_ms, volume=0.5, wave='square'):
    """Synthesise a simple tone as a pygame.mixer.Sound object."""
    n     = int(SAMPLE_RATE * duration_ms / 1000)
    t     = np.linspace(0, duration_ms / 1000, n, False)
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
    # Fade out last 20 % to prevent clicks
    fade = max(1, int(n * 0.2))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    stereo = np.column_stack([sig, sig])
    return pygame.sndarray.make_sound(stereo)


def _make_sweep(freq_start, freq_end, duration_ms, volume=0.5, wave='sine'):
    """Synthesise a frequency-sweep tone (e.g. rising fanfare)."""
    n    = int(SAMPLE_RATE * duration_ms / 1000)
    t    = np.linspace(0, duration_ms / 1000, n, False)
    freq = np.linspace(freq_start, freq_end, n)
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    if wave == 'sine':
        sig = np.sin(phase)
    else:
        sig = np.sign(np.sin(phase))
    fade = max(1, int(n * 0.15))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    stereo = np.column_stack([sig, sig])
    return pygame.sndarray.make_sound(stereo)


def _load_or_generate(key, generator_fn):
    """
    Return a pygame.mixer.Sound from a file if AUDIO_FILES[key] is set,
    otherwise call generator_fn() to synthesise one.
    """
    path = AUDIO_FILES.get(key)
    if path and os.path.isfile(path):
        snd = pygame.mixer.Sound(path)
        snd.set_volume(SFX_VOLUME)
        return snd
    return generator_fn()


def build_sounds():
    """
    Build all game sounds.  Returns a dict of pygame.mixer.Sound objects.
    Each entry falls back to a synthesised tone when no file is configured.
    """
    sounds = {}

    # Shoot — short high-pitched square blip
    sounds['shoot'] = _load_or_generate(
        'shoot',
        lambda: _make_tone(880, 55, volume=SFX_VOLUME * 0.6, wave='square'),
    )

    # Explosions — one per enemy type
    sounds['explode_0'] = _load_or_generate(
        'explode_0',
        lambda: _make_tone(180, 220, volume=SFX_VOLUME * 0.8, wave='noise'),
    )
    sounds['explode_1'] = _load_or_generate(
        'explode_1',
        lambda: _make_tone(320, 130, volume=SFX_VOLUME * 0.7, wave='noise'),
    )
    sounds['explode_2'] = _load_or_generate(
        'explode_2',
        lambda: _make_tone(90,  380, volume=SFX_VOLUME,       wave='noise'),
    )

    # Round-end fanfare — rising sweep chord
    sounds['round_end'] = _load_or_generate(
        'round_end',
        lambda: _make_sweep(220, 880, 700, volume=SFX_VOLUME * 0.9, wave='sine'),
    )

    # Low-time tick — short mid-range blip
    sounds['tick'] = _load_or_generate(
        'tick',
        lambda: _make_tone(520, 80, volume=SFX_VOLUME * 0.5, wave='square'),
    )

    return sounds


def start_bgm():
    """Start looping background music (file) or a simple generated drone."""
    path = AUDIO_FILES.get('bgm')
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(BGM_VOLUME)
        pygame.mixer.music.play(-1)
    else:
        # Synthesise a simple pulsing drone as BGM on a dedicated channel
        # Two detuned square waves layered for a retro space feel
        n       = int(SAMPLE_RATE * 2.0)          # 2-second loop
        t       = np.linspace(0, 2.0, n, False)
        wave_a  = np.sign(np.sin(2 * np.pi * 55  * t))   # low A
        wave_b  = np.sign(np.sin(2 * np.pi * 55.7 * t))  # slightly detuned
        pulse   = (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz volume pulse
        sig     = ((wave_a + wave_b) * 0.5 * pulse * BGM_VOLUME * 32767).astype(np.int16)
        stereo  = np.column_stack([sig, sig])
        bgm_snd = pygame.sndarray.make_sound(stereo)
        bgm_snd.play(loops=-1)


# ---------------------------------------------------------------------------
# Draw helpers (unchanged from previous version)
# ---------------------------------------------------------------------------

def _draw_bullet_trail(surf, bullet, player_color):
    trail = bullet.get('trail', [])
    n = len(trail)
    if n < 2:
        return
    for k in range(n - 1):
        alpha_ratio = k / n
        r = min(255, int(player_color[0] * alpha_ratio + 255 * (1 - alpha_ratio)))
        g = min(255, int(player_color[1] * alpha_ratio))
        b = min(255, int(player_color[2] * alpha_ratio))
        width_px = max(1, int(BULLET_RADIUS * alpha_ratio))
        pygame.draw.line(surf, (r, g, b),
                         (int(trail[k][0]),   int(trail[k][1])),
                         (int(trail[k+1][0]), int(trail[k+1][1])),
                         width_px)


def _draw_enemy_overlay(surf, en):
    speed_mult, points, ring_color, _ = ENEMY_TYPES[en['type']]
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
        color     = tuple(min(255, int(c * alpha / 255)) for c in p['color'])
        text_surf = popup_font.render(p['text'], True, color)
        surf.blit(text_surf, (int(p['x']) - text_surf.get_width() // 2, int(p['y'])))


def _draw_ghost_ship(surf, x, color):
    ghost_surf = pygame.Surface((SHIP_WIDTH, SHIP_HEIGHT), pygame.SRCALPHA)
    cx  = SHIP_WIDTH // 2
    pts = [(cx, 0), (0, SHIP_HEIGHT), (SHIP_WIDTH, SHIP_HEIGHT)]
    pygame.draw.polygon(ghost_surf, (*color, GHOST_ALPHA), pts)
    pygame.draw.polygon(ghost_surf, (*color, GHOST_ALPHA + 40), pts, 2)
    surf.blit(ghost_surf, (x, WINDOW_HEIGHT - SHIP_HEIGHT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    screen    = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hailo 3D Space Invaders")
    clock     = pygame.time.Clock()
    stars3d   = generate_stars_3d()
    game_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    # Fonts
    banner_font = pygame.font.Font(None, 72)
    popup_font  = pygame.font.Font(None, 36)

    # Audio
    sounds = build_sounds()
    start_bgm()

    colors = [GREEN, BLUE, YELLOW, PURPLE]
    bullet_outer_colors = [tuple(min(255, c + 80) for c in col) for col in colors]

    ship_positions    = [WINDOW_WIDTH // (MAX_PLAYERS + 1) * (i + 1) for i in range(MAX_PLAYERS)]
    position_histories = [deque([ship_positions[i]] * SMOOTHING_WINDOW, maxlen=SMOOTHING_WINDOW) for i in range(MAX_PLAYERS)]
    bullets           = [[] for _ in range(MAX_PLAYERS)]
    enemies           = []
    explosions        = []
    fireworks         = []
    score_popups      = []
    scores            = [0] * MAX_PLAYERS
    high_score        = load_high_score()

    start_time        = time.time()
    frame             = 0
    shoot_cooldowns   = [0] * MAX_PLAYERS
    banner_timer      = 0
    banner_y          = -100
    shake_timer       = 0
    last_tick_second  = -1   # tracks which second we last played the tick sound

    user_data   = PoseInvadersUserData()
    callback    = PoseInvadersCallback(user_data)
    app         = GStreamerPoseEstimationApp(callback, user_data)
    pose_thread = threading.Thread(target=app.run)
    pose_thread.daemon = True
    pose_thread.start()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # --- Keyboard fallback ---
        keys = pygame.key.get_pressed()
        kb_controls = [
            (pygame.K_a,    pygame.K_d),
            (pygame.K_LEFT, pygame.K_RIGHT),
        ]
        for i, (key_left, key_right) in enumerate(kb_controls):
            if i >= MAX_PLAYERS:
                break
            if keys[key_left]:
                position_histories[i].append(max(0, position_histories[i][-1] - 8))
            elif keys[key_right]:
                position_histories[i].append(min(WINDOW_WIDTH - SHIP_WIDTH, position_histories[i][-1] + 8))

        elapsed = time.time() - start_time
        t_left  = max(0, int(ROUND_TIME - elapsed))

        # --- Round end ---
        if t_left == 0:
            explosions.append({'x': WINDOW_WIDTH / 2, 'y': WINDOW_HEIGHT / 2, 't': 30})
            shake_timer = SHAKE_FRAMES
            sounds['round_end'].play()
            pygame.mixer.music.stop()   # silence BGM during fanfare
            if max(scores) > high_score:
                high_score = max(scores)
                save_high_score(high_score)
            scores = [0] * MAX_PLAYERS
            enemies.clear()
            for bl in bullets:
                bl.clear()
            score_popups.clear()
            start_time       = time.time()
            banner_timer     = INITIAL_BANNER_TIMER
            banner_y         = -100
            last_tick_second = -1
            # Restart BGM after the fanfare (approx 0.7 s)
            pygame.time.set_timer(pygame.USEREVENT + 1, 800, loops=1)

        # Restart BGM after fanfare delay
        for e in pygame.event.get(pygame.USEREVENT + 1):
            start_bgm()

        # --- Low-time countdown tick (once per second) ---
        if 0 < t_left <= LOW_TIME_THRESHOLD and t_left != last_tick_second:
            sounds['tick'].play()
            last_tick_second = t_left

        # --- Read pose positions ---
        for i in range(MAX_PLAYERS):
            try:
                head_x = user_data.position_queues[i].get_nowait()
                position_histories[i].append(head_x)
            except queue.Empty:
                pass

        ship_positions = [
            np.clip(int(sum(hist) / len(hist)) - SHIP_WIDTH // 2, 0, WINDOW_WIDTH - SHIP_WIDTH)
            for hist in position_histories
        ]

        # --- Shooting ---
        active_players = user_data.active_players
        for i in range(MAX_PLAYERS):
            if active_players[i]:
                shoot_cooldowns[i] -= 1
                if shoot_cooldowns[i] <= 0:
                    left_x  = ship_positions[i] + SHIP_WIDTH * 0.2
                    right_x = ship_positions[i] + SHIP_WIDTH * 0.8
                    y_b     = WINDOW_HEIGHT - SHIP_HEIGHT * 0.5
                    for bx in (left_x, right_x):
                        bullets[i].append({'x': bx, 'y': y_b, 'trail': [], 'color': colors[i]})
                    shoot_cooldowns[i] = SHOOT_INTERVAL_FRAMES
                    sounds['shoot'].play()

        # --- Move bullets ---
        for bullet_list in bullets:
            for b in bullet_list[:]:
                b['trail'].append((b['x'], b['y']))
                if len(b['trail']) > BULLET_TRAIL_LEN:
                    b['trail'].pop(0)
                b['y'] -= BULLET_SPEED
                if b['y'] < 0:
                    bullet_list.remove(b)

        # --- Spawn enemies ---
        if random.random() < ENEMY_SPAWN_CHANCE:
            etype = random.choices(range(len(ENEMY_TYPES)), weights=_ENEMY_TYPE_WEIGHTS)[0]
            enemies.append({
                'x':     random.randint(0, WINDOW_WIDTH - ENEMY_WIDTH),
                'y':     0,
                'type':  etype,
                'phase': random.uniform(0, math.pi * 2),
            })

        # --- Move enemies & collision ---
        for en in enemies[:]:
            speed_mult, points, ring_color, _ = ENEMY_TYPES[en['type']]
            en['y'] += ENEMY_SPEED * speed_mult
            drift   = math.sin(en['y'] * 0.02 + en['phase']) * ENEMY_DRIFT_SPEED
            en['x'] = float(np.clip(en['x'] + drift, 0, WINDOW_WIDTH - ENEMY_WIDTH))

            if en['y'] > WINDOW_HEIGHT:
                enemies.remove(en)
                continue

            cx  = int(en['x'] + ENEMY_WIDTH  // 2)
            cy  = int(en['y'] + ENEMY_HEIGHT // 2)
            hit = False
            for i, bullet_list in enumerate(bullets):
                for b in bullet_list[:]:
                    if math.hypot(b['x'] - cx, b['y'] - cy) < ENEMY_WIDTH / 2:
                        bullet_list.remove(b)
                        enemies.remove(en)
                        explosions.append({'x': cx, 'y': cy, 't': 15})
                        scores[i] += points
                        score_popups.append({
                            'x': cx, 'y': cy,
                            'text': f'+{points}',
                            'color': colors[i],
                            't': 40, 'max_t': 40,
                        })
                        # Play explosion sound matching enemy type
                        sounds[f'explode_{en["type"]}'].play()
                        hit = True
                        break
                if hit:
                    break

        # --- Age explosions ---
        for ex in explosions[:]:
            ex['t'] -= 1
            if ex['t'] <= 0:
                explosions.remove(ex)

        # --- Age score popups ---
        for p in score_popups[:]:
            p['y'] -= 1.2
            p['t'] -= 1
            if p['t'] <= 0:
                score_popups.remove(p)

        # --- Screen shake ---
        if shake_timer > 0:
            shake_timer -= 1

        # ================================================================
        # DRAW
        # ================================================================
        draw_background_3d(game_surf, stars3d)

        for i in range(MAX_PLAYERS):
            if not active_players[i]:
                _draw_ghost_ship(game_surf, ship_positions[i], colors[i])

        for i in range(MAX_PLAYERS):
            if active_players[i]:
                draw_ship(game_surf, ship_positions[i], WINDOW_HEIGHT - SHIP_HEIGHT, frame, colors[i])
                outer_col = bullet_outer_colors[i]
                inner_col = colors[i]
                for b in bullets[i]:
                    _draw_bullet_trail(game_surf, b, colors[i])
                    fh = BULLET_RADIUS * 6
                    bx, by = int(b['x']), int(b['y'])
                    br     = BULLET_RADIUS
                    pygame.draw.polygon(game_surf, outer_col,
                                        [(bx, by), (bx - br, by + fh), (bx + br, by + fh)])
                    pygame.draw.polygon(game_surf, inner_col,
                                        [(bx, by + int(fh * 0.4)), (bx - br // 2, by + fh), (bx + br // 2, by + fh)])

        for en in enemies:
            draw_enemy(game_surf, en['x'], en['y'])
            _draw_enemy_overlay(game_surf, en)

        for ex in explosions:
            draw_explosion(game_surf, ex['x'], ex['y'], ex['t'])

        _draw_score_popups(game_surf, score_popups, popup_font)

        # --- End-of-round banner ---
        if banner_timer > 0:
            banner_y = min(200, banner_y + BANNER_SPEED)
            progress = INITIAL_BANNER_TIMER - banner_timer
            wobble   = math.sin(progress / 10) * BANNER_WOBBLE_AMP
            text     = "KANGAN DIGITAL INITIATIVE"
            total    = banner_font.size(text)[0]
            start_x  = WINDOW_WIDTH / 2 - total / 2 + wobble
            for j, ch in enumerate(text):
                ch_surf = banner_font.render(ch, True, YELLOW)
                ch_x = start_x + banner_font.size(text[:j])[0]
                ch_y = banner_y + 30 + math.sin((progress + j * 5) / 5) * BANNER_WAVE_AMP
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

        # --- HUD ---
        for i in range(MAX_PLAYERS):
            draw_text(game_surf, f"P{i+1} SCORE: {scores[i]}", 10, 10 + i * 30, colors[i])
        draw_text(game_surf, f"HIGH: {high_score}", 10, 10 + MAX_PLAYERS * 30, GREEN)

        if t_left <= LOW_TIME_THRESHOLD and frame % 20 < 10:
            timer_color = (255, 50, 50)
        else:
            timer_color = WHITE
        draw_text(game_surf, f"TIME: {t_left}", WINDOW_WIDTH - 150, 10, timer_color)

        # --- Blit with shake ---
        if shake_timer > 0:
            amp = int(SHAKE_AMPLITUDE * shake_timer / SHAKE_FRAMES)
            sx  = random.randint(-amp, amp)
            sy  = random.randint(-amp, amp)
        else:
            sx, sy = 0, 0
        screen.fill((0, 0, 0))
        screen.blit(game_surf, (sx, sy))

        pygame.display.flip()
        clock.tick(60)
        frame += 1

    save_high_score(high_score)
    pygame.mixer.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
