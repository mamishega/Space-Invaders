# Hailo 3D Space Invaders
### Raspberry Pi 5 + Hailo AI HAT+ Setup Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Prerequisites](#software-prerequisites)
4. [OS Setup](#os-setup)
5. [Hailo AI HAT+ Installation](#hailo-ai-hat-installation)
6. [Camera Setup](#camera-setup)
7. [Python Dependencies](#python-dependencies)
8. [Pose Estimation Model](#pose-estimation-model)
9. [Project Structure](#project-structure)
10. [Configuration](#configuration)
11. [Running the Game](#running-the-game)
12. [Controls](#controls)
13. [Gameplay](#gameplay)
14. [Troubleshooting](#troubleshooting)
15. [Performance Tips](#performance-tips)
16. [Credits](#credits)

---

## Overview

**Hailo 3D Space Invaders** is a multiplayer, camera-controlled remake of the classic Space Invaders arcade game. Instead of a joystick, players control their ships by **moving their head or body left and right** in front of a camera — no buttons, no controllers.

Key features:
- Up to **4 simultaneous players**, each tracked independently
- Real-time **pose estimation** powered by the Hailo AI HAT+ (Hailo-8L, 13 TOPS)
- **3D starfield** background with animated explosions and fireworks
- **Three enemy types** — Normal, Fast, and Tank — with distinct speeds and point values
- **Procedural audio** — all sounds generated at runtime, no audio files required
- High score persistence between sessions
- Built with **Python, pygame, GStreamer, and the Hailo SDK**

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Raspberry Pi 5 | 4 GB or 8 GB RAM recommended |
| Hailo AI HAT+ | Hailo-8L (13 TOPS), attaches via M.2 PCIe on RPi5 |
| Camera | Raspberry Pi Camera Module 3 (or any compatible CSI camera) |
| MicroSD Card | 32 GB+, Class 10 / A1 or faster |
| Display | HDMI monitor (1080p or lower recommended) |
| Input | USB keyboard + mouse (for setup) |
| Power Supply | Official Raspberry Pi 5 — 27 W USB-C PSU |
| Cooling (optional) | Active fan or heatsink — recommended for sustained play |

> The Hailo AI HAT+ connects to the **top M.2 slot** on the Raspberry Pi 5 via PCIe. No soldering required — it is a plug-in module.

---

## Software Prerequisites

- Raspberry Pi OS **Bookworm** (64-bit, Desktop) — minimum required OS version
- Python **3.11+**
- HailoRT runtime (`hailo-all` package)
- `hailo_apps_infra` Python package
- GStreamer 1.0 with Hailo plugins
- pygame 2.x
- numpy

---

## OS Setup

### 1. Flash Raspberry Pi OS

Download and flash **Raspberry Pi OS Bookworm (64-bit, Desktop)** using the official Raspberry Pi Imager:

```
https://www.raspberrypi.com/software/
```

### 2. Enable PCIe Gen 3 (required for Hailo HAT+)

Open the boot config file:

```bash
sudo nano /boot/firmware/config.txt
```

Add the following line at the bottom:

```
dtparam=pciex1_gen=3
```

Save and close (`Ctrl+X → Y → Enter`).

### 3. Enable the Camera

```bash
sudo raspi-config
```

Navigate to: **Interface Options → Camera → Enable**

### 4. Update the System

```bash
sudo apt update && sudo apt full-upgrade -y
```

### 5. Reboot

```bash
sudo reboot
```

---

## Hailo AI HAT+ Installation

### Physical Installation

1. Power off the Raspberry Pi 5 completely
2. Attach the Hailo AI HAT+ to the **top M.2 PCIe slot** on the RPi5
3. Secure it with the included standoffs and screw
4. Power the Pi back on

### Install the Hailo Software Stack

```bash
sudo apt install hailo-all -y
```

This installs HailoRT, the PCIe driver, GStreamer Hailo plugins, and all supporting tools.

### Verify the Installation

```bash
hailortcli fw-control identify
```

Expected output (example):

```
Hailo-8L Device:
  Firmware Version: 4.x.x
  Serial Number:    HXXXXXXXXX
  ...
```

### Install the Python Runtime

```bash
pip install hailort
```

### Install hailo_apps_infra

```bash
git clone https://github.com/hailo-ai/hailo-apps-infra
cd hailo-apps-infra
pip install -e .
```

---

## Camera Setup

### Connect the Camera

Connect the **Raspberry Pi Camera Module 3** to the CSI/MIPI camera port on the RPi5. Ensure the ribbon cable is fully inserted with the contacts facing the correct direction.

### Verify the Camera is Detected

```bash
libcamera-hello --list-cameras
```

You should see at least one camera listed (e.g., `imx708`).

### Test the Camera Preview

```bash
libcamera-hello -t 5000
```

This opens a 5-second live preview. If the preview appears, the camera is working correctly.

---

## Python Dependencies

Install system packages:

```bash
sudo apt install -y \
  python3-pygame \
  python3-numpy \
  python3-gi \
  python3-gi-cairo \
  gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good
```

Install Python packages:

```bash
pip install pygame numpy
```

---

## Pose Estimation Model

The game uses a **YOLOv8 pose estimation model** compiled for the Hailo-8L chip (`.hef` format).

### Download the Model

```bash
# Create the models directory inside the project
mkdir -p ~/Space_invaders/models

# Download via hailo_apps_infra helper (if available)
python3 -c "from hailo_apps_infra import download_resources; download_resources()"
```

Alternatively, download `yolov8s_pose.hef` manually from the Hailo Model Zoo and place it at:

```
Space_invaders/models/yolov8s_pose.hef
```

`hailo_apps_infra` will also search `~/.hailo/models/` automatically.

---

## Project Structure

```
Space_invaders/
├── Space_Invaders_16_Multiplayer_V6.py   # Main game
├── generate_changelog.py                 # Generates changelog.png
├── changelog.png                         # Visual changelog infographic
├── models/
│   └── yolov8s_pose.hef                  # Hailo-8L pose estimation model
├── sounds/                               # Optional — real audio files
│   ├── shoot.wav
│   ├── explode_small.wav
│   ├── explode_fast.wav
│   ├── explode_big.wav
│   ├── fanfare.wav
│   ├── tick.wav
│   └── bgm.ogg
├── .claude/
│   └── launch.json                       # Dev launch configurations
└── README.md                             # This file
```

> The `sounds/` folder is **optional**. If no audio files are present, the game automatically synthesises all sounds at startup using numpy — no downloads required.

---

## Configuration

All tunable constants are defined near the top of `Space_Invaders_16_Multiplayer_V6.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_PLAYERS` | `4` | Number of simultaneous players (1–4) |
| `WINDOW_WIDTH` | — | Display width in pixels |
| `WINDOW_HEIGHT` | — | Display height in pixels |
| `ROUND_TIME` | — | Round duration in seconds |
| `SMOOTHING_WINDOW` | — | Head-tracking smoothness (higher = smoother but slower) |
| `SHOOT_INTERVAL_FRAMES` | — | Frames between auto-fire bursts |
| `ENEMY_SPAWN_CHANCE` | `0.02` | Probability per frame a new enemy spawns |
| `LOW_TIME_THRESHOLD` | `10` | Seconds remaining before timer flashes red |
| `BGM_VOLUME` | `0.35` | Background music volume (0.0–1.0) |
| `SFX_VOLUME` | `0.7` | Sound effects volume (0.0–1.0) |

### Using Real Audio Files

To replace the generated tones with real `.wav` / `.ogg` files, edit the `AUDIO_FILES` dictionary:

```python
AUDIO_FILES = {
    'shoot':     'sounds/shoot.wav',
    'explode_0': 'sounds/explode_small.wav',
    'explode_1': 'sounds/explode_fast.wav',
    'explode_2': 'sounds/explode_big.wav',
    'round_end': 'sounds/fanfare.wav',
    'tick':      'sounds/tick.wav',
    'bgm':       'sounds/bgm.ogg',
}
```

Any entry set to `None` will fall back to a generated tone automatically.

---

## Virtual Environment (Hailo)

The Hailo RPi5 examples repo ships with its own pre-configured virtual environment. **Use this — do not create a separate venv.**

### First-time setup
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples ~/hailo-rpi5-examples
cd ~/hailo-rpi5-examples
source setup_env.sh   # creates venv_hailo_rpi5_examples and sets all env vars
```

### Copy the game into the tests folder
```bash
cp ~/Space_invaders/Space_Invaders_16_Multiplayer_V6.py \
   ~/hailo-rpi5-examples/tests/Space_Invaders_16_Multiplayer_V5.py
```

---

## Running the Game

1. Connect your display, camera, and keyboard to the Raspberry Pi 5
2. Open a terminal
3. Navigate to the project directory:

```bash
cd ~/Space_invaders
```

4. Source the Hailo environment and run the game:

**USB camera (webcam):**
```bash
source ~/hailo-rpi5-examples/setup_env.sh && \
python ~/hailo-rpi5-examples/tests/Space_Invaders.py --input usb
```

**Raspberry Pi Camera Module:**
```bash
source ~/hailo-rpi5-examples/setup_env.sh && \
python ~/hailo-rpi5-examples/tests/Space_Invaders.py --input rpi
```

5. Players stand in front of the camera — the game will detect them automatically and assign each person a ship

> If running over **SSH**, set the display variable first:
> ```bash
> DISPLAY=:0 source ~/hailo-rpi5-examples/setup_env.sh && \
> python ~/hailo-rpi5-examples/tests/Space_Invaders_16_Multiplayer_V5.py --input usb
> ```

---

## Controls

### Camera Controls (Hailo Pose Estimation)

| Action | How |
|---|---|
| Move ship left | Lean / step your body to the left |
| Move ship right | Lean / step your body to the right |
| Shoot | Automatic — no action needed |

### Keyboard Fallback (no camera required)

Useful for testing without the Hailo hardware:

| Player | Left | Right |
|---|---|---|
| Player 1 | `A` | `D` |
| Player 2 | `LEFT ARROW` | `RIGHT ARROW` |

Press `Ctrl+C` in the terminal or close the window to quit.

---

## Gameplay

- Up to **4 players** are tracked and assigned ships at the bottom of the screen
- Each player's ship fires **automatically** — just focus on moving
- Destroy enemies before they reach the bottom of the screen
- The round ends when the timer runs out — scores reset and a new round begins
- The **high score** is saved between sessions

### Enemy Types

| Type | Ring | Speed | Points | Behaviour |
|---|---|---|---|---|
| Normal | None | 1× | 10 | Straight descent |
| Fast | Red | 2× | 20 | Fast, weaves side to side |
| Tank | Blue | 0.5× | 30 | Slow, difficult to intercept |

### HUD

| Element | Description |
|---|---|
| `P1 SCORE` … `P4 SCORE` | Live score per player (colour-coded) |
| `HIGH` | All-time high score |
| `TIME` | Seconds remaining in the round (flashes red ≤ 10 s) |
| Floating `+10/+20/+30` | Score popup on each kill |
| Ghost ship outline | Shows position of inactive / undetected players |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No Hailo device found` | Check the HAT+ is seated firmly; run `hailortcli fw-control identify` |
| `Camera not found` | Run `libcamera-hello --list-cameras`; check the CSI ribbon cable is fully inserted |
| `ImportError: hailo` | Reinstall: `sudo apt install --reinstall hailo-all` |
| `pygame display error` | If using SSH: `DISPLAY=:0 python3 Space_Invaders_16_Multiplayer_V6.py` |
| Low frame rate | Enable PCIe Gen 3 (see OS Setup); add cooling to prevent throttling |
| GStreamer errors | Verify Hailo plugins: `gst-inspect-1.0 hailotools` |
| Audio issues | Test mixer: `python3 -c "import pygame; pygame.mixer.init(); print('OK')"` |
| No pose detection | Ensure good lighting; players should be fully visible to the camera |

---

## Performance Tips

- **Enable PCIe Gen 3** — add `dtparam=pciex1_gen=3` to `/boot/firmware/config.txt` (single biggest improvement)
- **Set CPU governor to performance mode:**
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
- **Use active cooling** — the RPi5 and Hailo HAT+ both throttle under sustained load without a fan or heatsink
- **Close background apps** — web browsers and file managers consume RAM and CPU
- **Use 1080p or lower resolution** — higher resolutions reduce pygame rendering performance
- **Run headlessly over SSH** with `DISPLAY=:0` if the desktop environment is consuming resources

---

