"""
main.py - Self-Driving Car AI Simulation with Official MotoGP Circuits
Interactive Pygame application with:
  - 9 Authentic MotoGP World Championship Tracks with varied difficulty
  - Auto-begin new generation when any car reaches the Race End Point (Finish Line)
  - Live Neural Network visualizer
  - Fitness history graph
  - Toast notifications
  - Fast-forward training (1x/2x/5x/10x)
  - Direct circuit selection (Keys 1-9)
  - Manual driving mode
  - Save/Load brain to JSON
"""
import sys
import os
import time
import math
import pygame
from track import Track, MOTOGP_CIRCUITS
from population import Population
from car import Car

# ── Constants ───────────────────────────────────────────────────────────────
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 768
FPS = 60

# Colors
C_BG = (16, 18, 24)
C_HUD_BG = (14, 17, 24)
C_HUD_BORDER = (45, 55, 75)
C_TEXT = (225, 235, 245)
C_TEXT_DIM = (120, 140, 165)
C_GOLD = (255, 215, 0)
C_CYAN = (0, 240, 255)
C_GREEN = (0, 215, 120)
C_ORANGE = (255, 160, 40)
C_RED = (255, 70, 70)
C_PURPLE = (220, 90, 255)
C_PANEL_BG = (19, 23, 32)
C_PANEL_BORDER = (55, 70, 95)

DIFF_COLORS = {
    "Medium": C_GREEN,
    "Hard": C_ORANGE,
    "Very Hard": C_RED,
    "Extreme": C_PURPLE
}


# ── Toast notification system ───────────────────────────────────────────────
class ToastManager:
    def __init__(self):
        self.messages = []  # list of (text, expire_time, color)

    def show(self, text, duration=2.5, color=C_CYAN):
        self.messages.append((text, time.time() + duration, color))

    def draw(self, surface, font):
        now = time.time()
        self.messages = [(t, e, c) for t, e, c in self.messages if e > now]
        y = 66
        for text, expire, color in self.messages:
            alpha = min(1.0, (expire - now) / 0.5)
            txt = font.render(text, True, color)
            pad = 8
            bg = pygame.Surface((txt.get_width() + pad * 2, txt.get_height() + pad), pygame.SRCALPHA)
            bg.fill((*C_PANEL_BG, int(210 * alpha)))
            bx = SCREEN_WIDTH // 2 - bg.get_width() // 2
            surface.blit(bg, (bx, y))
            surface.blit(txt, (bx + pad, y + pad // 2))
            y += txt.get_height() + pad + 4


# ── Neural Network Visualizer ──────────────────────────────────────────────
def draw_network_panel(surface, brain, x, y, w, h, font):
    """Draws a real-time neural network activation visualizer."""
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((*C_PANEL_BG, 220))
    surface.blit(bg, (x, y))
    pygame.draw.rect(surface, C_PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

    title = font.render("NEURAL NETWORK", True, (160, 185, 215))
    surface.blit(title, (x + 10, y + 6))

    if brain is None or not brain.last_activations:
        return

    num_layers = len(brain.layer_sizes)
    col_gap = (w - 60) / max(1, num_layers - 1)

    node_pos = []
    for l_idx, count in enumerate(brain.layer_sizes):
        nodes = []
        cx = x + 30 + l_idx * col_gap
        row_gap = (h - 40) / (count + 1)
        for n_idx in range(count):
            ny_ = y + 28 + (n_idx + 1) * row_gap
            nodes.append((cx, ny_))
        node_pos.append(nodes)

    # Connections
    for l in range(len(brain.weights)):
        for i in range(len(brain.weights[l])):
            for j in range(len(brain.weights[l][0])):
                wt = brain.weights[l][i][j]
                p1 = node_pos[l][i]
                p2 = node_pos[l + 1][j]
                intensity = min(255, int(abs(wt) * 50))
                c = (0, intensity, int(intensity * 0.7)) if wt > 0 else (intensity, 0, 0)
                lw = max(1, min(3, int(abs(wt) * 1.2)))
                pygame.draw.line(surface, c, p1, p2, lw)

    # Nodes
    input_labels = ["-60", "-30", "0", "+30", "+60", "Spd"]
    output_labels = ["Str", "Thr"]

    for l_idx, layer in enumerate(node_pos):
        acts = brain.last_activations[l_idx] if l_idx < len(brain.last_activations) else []
        for n_idx, (nx_, ny_) in enumerate(layer):
            val = acts[n_idx] if n_idx < len(acts) else 0.0
            if val >= 0:
                nc = (int(40 + val * 180), int(140 + val * 115), int(80 + val * 175))
            else:
                nc = (int(170 + abs(val) * 85), int(50 * (1 - abs(val))), 60)

            r = 5 if (l_idx == 0 or l_idx == num_layers - 1) else 4
            pygame.draw.circle(surface, nc, (int(nx_), int(ny_)), r)
            pygame.draw.circle(surface, (200, 215, 235), (int(nx_), int(ny_)), r, 1)

            if l_idx == 0 and n_idx < len(input_labels):
                lbl = font.render(input_labels[n_idx], True, C_TEXT_DIM)
                surface.blit(lbl, (nx_ - 24, ny_ - 5))
            elif l_idx == num_layers - 1 and n_idx < len(output_labels):
                lbl = font.render(output_labels[n_idx], True, C_TEXT_DIM)
                surface.blit(lbl, (nx_ + 8, ny_ - 5))


# ── Fitness History Graph ───────────────────────────────────────────────────
def draw_fitness_graph(surface, history, x, y, w, h, font):
    """Draws a mini line-chart of best fitness per generation."""
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((*C_PANEL_BG, 220))
    surface.blit(bg, (x, y))
    pygame.draw.rect(surface, C_PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

    title = font.render("FITNESS HISTORY", True, (160, 185, 215))
    surface.blit(title, (x + 10, y + 6))

    if len(history) < 2:
        msg = font.render("Waiting for data...", True, C_TEXT_DIM)
        surface.blit(msg, (x + w // 2 - msg.get_width() // 2, y + h // 2 - 6))
        return

    gx = x + 30
    gy = y + 28
    gw = w - 40
    gh = h - 42

    pygame.draw.line(surface, (55, 68, 88), (gx, gy + gh), (gx + gw, gy + gh), 1)
    pygame.draw.line(surface, (55, 68, 88), (gx, gy), (gx, gy + gh), 1)

    display = history[-30:]
    max_fit = max(display) if display else 1.0
    if max_fit == 0:
        max_fit = 1.0

    points = []
    for i, fit in enumerate(display):
        px = gx + (i / max(1, len(display) - 1)) * gw
        py = gy + gh - (fit / max_fit) * gh
        points.append((int(px), int(py)))

    if len(points) >= 2:
        pygame.draw.lines(surface, C_GREEN, False, points, 2)

    for pt in points:
        pygame.draw.circle(surface, C_CYAN, pt, 2)

    top_lbl = font.render(f"{int(max_fit)}", True, C_TEXT_DIM)
    surface.blit(top_lbl, (gx - 28, gy - 4))
    gen_lbl = font.render(f"Gen {len(history)}", True, C_TEXT_DIM)
    surface.blit(gen_lbl, (gx + gw - gen_lbl.get_width(), gy + gh + 3))


# ── HUD ────────────────────────────────────────────────────────────────────
def draw_hud(surface, pop, track, speed_mult, manual_mode, track_idx, total_tracks, font, small_font):
    """Top info banner and bottom keybindings helper."""
    hud = pygame.Surface((SCREEN_WIDTH, 48), pygame.SRCALPHA)
    hud.fill((*C_HUD_BG, 235))
    surface.blit(hud, (0, 0))
    pygame.draw.line(surface, C_HUD_BORDER, (0, 48), (SCREEN_WIDTH, 48), 1)

    alive = pop.count_alive()
    diff_color = DIFF_COLORS.get(track.difficulty, C_TEXT)
    leader_cps = pop.best_car.checkpoints_passed if pop.best_car else 0
    total_cps = len(track.checkpoints)

    items = [
        (f"GEN {pop.generation}", C_TEXT),
        (f"Alive: {alive}/{pop.size}", C_GREEN if alive > pop.size // 3 else C_RED),
        (f"End Point: {leader_cps}/{total_cps} CPs", C_CYAN),
        (f"Best: {int(pop.current_best_fitness)}", C_TEXT),
        (f"Record: {int(pop.all_time_best_fitness)}", C_GOLD),
        (f"Speed: {speed_mult}x", C_CYAN),
        (f"[{track_idx + 1}/{total_tracks}] {track.name}", C_GOLD),
        (f"★ {track.difficulty.upper()}", diff_color),
        ("MANUAL MODE" if manual_mode else "AI EVOLUTION", C_CYAN if manual_mode else C_GREEN),
    ]

    ox = 14
    for text, color in items:
        ts = font.render(text, True, color)
        surface.blit(ts, (ox, 14))
        ox += ts.get_width() + 16

    # Bottom helper bar
    bbar = pygame.Surface((SCREEN_WIDTH, 28), pygame.SRCALPHA)
    bbar.fill((*C_HUD_BG, 210))
    surface.blit(bbar, (0, SCREEN_HEIGHT - 28))
    help_txt = "[Auto Next Gen on Race End Point]  [1-9 / T / Y] Tracks  [F] Speed  [SPACE] Skip Gen  [M] Manual  [R] Rays  [S/L] Save/Load"
    hs = small_font.render(help_txt, True, C_TEXT_DIM)
    surface.blit(hs, (SCREEN_WIDTH // 2 - hs.get_width() // 2, SCREEN_HEIGHT - 24))


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_caption("Self-Driving Car AI — MotoGP Championship Edition")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Segoe UI", 13, bold=True)
    small_font = pygame.font.SysFont("Segoe UI", 12)

    toasts = ToastManager()

    # Track selection
    track_names = list(MOTOGP_CIRCUITS.keys())
    track_idx = 0
    track = Track(track_names[track_idx])

    # Population
    pop_size = 50
    pop = Population(pop_size, track.start_x, track.start_y, track.start_angle)

    # Manual test car
    manual_car = Car(track.start_x, track.start_y, track.start_angle)

    # Settings
    speed_mults = [1, 2, 5, 10]
    speed_idx = 1      # Start at 2x speed for faster training!
    manual_mode = False
    show_rays = 1      # 0=off, 1=best only, 2=all
    show_cps = False

    walls = track.get_all_walls()

    toasts.show(f"🏁 MotoGP: {track.name} [{track.difficulty}]", 3.0, C_GOLD)
    toasts.show("When a car reaches the Race End Point, new gen begins!", 4.0, C_GREEN)

    def switch_to_track(new_idx):
        nonlocal track, track_idx, walls, pop, manual_car
        track_idx = new_idx % len(track_names)
        track = Track(track_names[track_idx])
        walls = track.get_all_walls()
        
        # Keep learned brains & current generation! Transfer cars to the new track's start line.
        pop.transfer_to_track(track.start_x, track.start_y, track.start_angle)
        manual_car = Car(track.start_x, track.start_y, track.start_angle)
        
        d_col = DIFF_COLORS.get(track.difficulty, C_GOLD)
        toasts.show(f"[{track_idx + 1}/9] {track.name} — {track.difficulty.upper()} (Gen {pop.generation} Kept!)", 3.0, d_col)
        if track.description:
            toasts.show(track.description, 3.5, C_TEXT_DIM)

    running = True
    while running:
        speed_mult = speed_mults[speed_idx]

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Number keys 1-9 to jump directly to any circuit
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    target_idx = event.key - pygame.K_1
                    if target_idx < len(track_names):
                        switch_to_track(target_idx)

                elif event.key == pygame.K_t:
                    switch_to_track(track_idx + 1)

                elif event.key == pygame.K_y:
                    switch_to_track(track_idx - 1)

                elif event.key == pygame.K_f:
                    speed_idx = (speed_idx + 1) % len(speed_mults)
                    toasts.show(f"Speed: {speed_mults[speed_idx]}x", 1.5)

                elif event.key == pygame.K_SPACE:
                    pop.natural_selection(track.checkpoints)
                    toasts.show(f"Forced evolution → Gen {pop.generation}", 1.5, C_GOLD)

                elif event.key == pygame.K_m:
                    manual_mode = not manual_mode
                    if manual_mode:
                        manual_car = Car(track.start_x, track.start_y, track.start_angle)
                        toasts.show("Manual mode ON — drive with WASD / Arrows", 2.5, C_CYAN)
                    else:
                        toasts.show("AI Evolution mode", 1.5, C_GREEN)

                elif event.key == pygame.K_r:
                    show_rays = (show_rays + 1) % 3
                    names = ["Off", "Best car", "All cars"]
                    toasts.show(f"Sensor rays: {names[show_rays]}", 1.5)

                elif event.key == pygame.K_c:
                    show_cps = not show_cps
                    toasts.show(f"Checkpoints: {'ON' if show_cps else 'OFF'}", 1.2)

                elif event.key == pygame.K_n:
                    pop.reset()
                    toasts.show("Population reset to Gen 1!", 1.5, C_RED)

                elif event.key == pygame.K_s:
                    save_dir = os.path.dirname(os.path.abspath(__file__))
                    fp = os.path.join(save_dir, "best_car_brain.json")
                    if pop.save_best_brain(fp):
                        toasts.show("Brain saved to best_car_brain.json", 2.0, C_GREEN)
                    else:
                        toasts.show("Nothing to save yet", 1.5, C_RED)

                elif event.key == pygame.K_l:
                    save_dir = os.path.dirname(os.path.abspath(__file__))
                    fp = os.path.join(save_dir, "best_car_brain.json")
                    try:
                        pop.load_brain_for_all(fp)
                        toasts.show("Brain loaded & spread to fleet!", 2.0, C_GREEN)
                    except FileNotFoundError:
                        toasts.show("No saved brain found", 1.5, C_RED)
                    except Exception as e:
                        toasts.show(f"Load error: {e}", 2.0, C_RED)

        # ── Physics (multiple sub-steps for speed multiplier) ───────────
        keys = pygame.key.get_pressed()

        for _ in range(speed_mult):
            if manual_mode:
                manual_car.update(walls, track.checkpoints, manual=True, keys=keys)
                if manual_car.has_finished_race:
                    toasts.show("🏁 YOU REACHED THE RACE END POINT! Lap Complete!", 3.0, C_GOLD)
                    manual_car = Car(track.start_x, track.start_y, track.start_angle)
                elif not manual_car.is_alive:
                    manual_car = Car(track.start_x, track.start_y, track.start_angle)
            else:
                # Update population & check for race end point completion
                winner = pop.update(walls, track.checkpoints)
                if winner is not None:
                    # Car has hit the Race End Point!
                    lap_time_s = winner.total_frames_alive / FPS
                    toasts.show(f"🏁 RACE END POINT REACHED! Lap in {lap_time_s:.1f}s!", 3.5, C_GOLD)
                    toasts.show(f"Beginning Generation {pop.generation + 1} with winner...", 2.5, C_GREEN)
                    pop.natural_selection(track.checkpoints)
                    break
                elif pop.is_extinct():
                    # All cars crashed or timed out
                    pop.natural_selection(track.checkpoints)
                    break

        # ── Rendering ───────────────────────────────────────────────────
        screen.fill(C_BG)
        track.draw(screen, show_checkpoints=show_cps, font=small_font)

        if manual_mode:
            manual_car.draw(screen, draw_sensors=(show_rays > 0))
        else:
            for car in pop.cars:
                if car != pop.best_car:
                    car.draw(screen, draw_sensors=(show_rays == 2))
            if pop.best_car:
                pop.best_car.draw(screen, draw_sensors=(show_rays >= 1))

        # HUD
        draw_hud(screen, pop, track, speed_mult, manual_mode, track_idx, len(track_names), font, small_font)

        # NN Visualizer (right side)
        brain = manual_car.brain if manual_mode else (pop.best_car.brain if pop.best_car else None)
        draw_network_panel(screen, brain, SCREEN_WIDTH - 290, 58, 275, 210, small_font)

        # Fitness graph (below NN panel)
        draw_fitness_graph(screen, pop.history, SCREEN_WIDTH - 290, 278, 275, 150, small_font)

        # Toasts
        toasts.draw(screen, font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
