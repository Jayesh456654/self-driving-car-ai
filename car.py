"""
car.py - Car physics, raycasting sensors, collision detection, and fitness.
Includes race end point detection (lap completion).
"""
import math
import pygame
from geometry import line_intersection, distance
from neural_network import NeuralNetwork


class Car:
    # NN architecture: 5 sensors + 1 speed → 10 hidden → 4 hidden → 2 outputs
    NN_LAYERS = [6, 10, 4, 2]

    def __init__(self, x, y, angle=0.0, brain=None):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.angle = angle  # degrees

        # Dimensions
        self.length = 24
        self.width = 12

        # Physics — tuned so cars accelerate and hold speed cleanly
        self.speed = 0.0
        self.max_speed = 5.5
        self.min_speed = 1.2          # AI always creeps forward
        self.acceleration = 0.18
        self.friction = 0.97
        self.turn_rate = 3.8          # deg / frame

        # Sensor Rays: 5 rays at -60, -30, 0, 30, 60 degrees relative to car heading
        self.ray_angles = [-60, -30, 0, 30, 60]
        self.max_ray_dist = 200.0
        self.sensor_readings = [1.0] * len(self.ray_angles)
        self.sensor_intersections = [None] * len(self.ray_angles)

        # State & Fitness
        self.is_alive = True
        self.death_reason = ""
        self.checkpoints_passed = 0
        self.current_checkpoint_idx = 0
        self.frames_since_checkpoint = 0
        self.total_frames_alive = 0
        self.total_distance = 0.0
        self.fitness = 0.0
        self.is_best = False
        self.has_finished_race = False  # Set to True when car reaches the race end point

        # Brain
        self.brain = brain if brain is not None else NeuralNetwork(self.NN_LAYERS)

    def reset_to_start(self, x, y, angle):
        """Resets car position and state to starting line while preserving its trained neural network brain."""
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.angle = angle
        self.speed = 0.0
        self.is_alive = True
        self.has_finished_race = False
        self.death_reason = ""
        self.checkpoints_passed = 0
        self.current_checkpoint_idx = 0
        self.frames_since_checkpoint = 0
        self.total_frames_alive = 0
        self.total_distance = 0.0
        self.fitness = 0.0
        self.sensor_readings = [1.0] * len(self.ray_angles)
        self.sensor_intersections = [None] * len(self.ray_angles)

    # ── Geometry helpers ────────────────────────────────────────────────────
    def get_corners(self):
        rad = math.radians(self.angle)
        hl = self.length / 2.0
        hw = self.width / 2.0
        offsets = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return [
            (self.x + ox * cos_a - oy * sin_a,
             self.y + ox * sin_a + oy * cos_a)
            for ox, oy in offsets
        ]

    def _edges(self, corners):
        return [
            (corners[i], corners[(i + 1) % 4])
            for i in range(4)
        ]

    # ── Sensors ─────────────────────────────────────────────────────────────
    def update_sensors(self, walls):
        self.sensor_readings = []
        self.sensor_intersections = []

        for rel_angle in self.ray_angles:
            ray_rad = math.radians(self.angle + rel_angle)
            ray_end = (
                self.x + math.cos(ray_rad) * self.max_ray_dist,
                self.y + math.sin(ray_rad) * self.max_ray_dist,
            )

            closest_dist = self.max_ray_dist
            closest_point = ray_end

            for wall in walls:
                hit = line_intersection((self.x, self.y), ray_end, wall[0], wall[1])
                if hit is not None:
                    _, _, d = hit
                    if d < closest_dist:
                        closest_dist = d
                        closest_point = (hit[0], hit[1])

            self.sensor_readings.append(closest_dist / self.max_ray_dist)
            self.sensor_intersections.append(closest_point)

    # ── Collision ───────────────────────────────────────────────────────────
    def check_collision(self, walls):
        for edge in self._edges(self.get_corners()):
            for wall in walls:
                if line_intersection(edge[0], edge[1], wall[0], wall[1]) is not None:
                    return True
        return False

    # ── Checkpoints & Race End Point ─────────────────────────────────────────
    def check_checkpoint(self, checkpoints):
        target = (self.current_checkpoint_idx + 1) % len(checkpoints)
        cp = checkpoints[target]
        if line_intersection(
            (self.prev_x, self.prev_y), (self.x, self.y), cp[0], cp[1]
        ) is not None:
            self.checkpoints_passed += 1
            self.current_checkpoint_idx = target
            self.frames_since_checkpoint = 0

            # When car crosses all checkpoints and reaches the race end point / finish line
            if self.checkpoints_passed >= len(checkpoints):
                self.has_finished_race = True
                return "finished"

            return True
        return False

    # ── Neural Network decision ─────────────────────────────────────────────
    def think_and_drive(self):
        if not self.is_alive or self.has_finished_race:
            return
        inputs = self.sensor_readings + [self.speed / self.max_speed]
        outputs = self.brain.forward(inputs)

        # Output 0: steering [-1..1]
        self.angle += outputs[0] * self.turn_rate

        # Output 1: throttle [-1..1]
        throttle = outputs[1]
        if throttle > 0:
            self.speed += throttle * self.acceleration
        else:
            self.speed += throttle * self.acceleration * 0.5

    # ── Manual driving ──────────────────────────────────────────────────────
    def manual_drive(self, keys):
        if not self.is_alive or self.has_finished_race:
            return
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= self.turn_rate
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += self.turn_rate
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.speed += self.acceleration
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.speed -= self.acceleration * 0.6

    # ── Main update ─────────────────────────────────────────────────────────
    def update(self, walls, checkpoints, manual=False, keys=None):
        if not self.is_alive or self.has_finished_race:
            return

        self.prev_x = self.x
        self.prev_y = self.y
        self.update_sensors(walls)

        if manual and keys is not None:
            self.manual_drive(keys)
        else:
            self.think_and_drive()

        # Apply friction and enforce minimum forward speed for AI cars
        self.speed *= self.friction
        if not manual:
            self.speed = max(self.min_speed, min(self.max_speed, self.speed))
        else:
            self.speed = max(0.0, min(self.max_speed, self.speed))

        # Move
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        self.x += dx
        self.y += dy

        self.total_distance += math.hypot(dx, dy)
        self.total_frames_alive += 1
        self.frames_since_checkpoint += 1

        # Collision with walls
        if self.check_collision(walls):
            self.is_alive = False
            self.death_reason = "crashed"
            return

        # Checkpoint check
        self.check_checkpoint(checkpoints)

        # Timeout (eliminates stuck cars)
        timeout = 220 if self.checkpoints_passed < 3 else 170
        if self.frames_since_checkpoint > timeout:
            self.is_alive = False
            self.death_reason = "timeout"

    # ── Fitness ─────────────────────────────────────────────────────────────
    def calculate_fitness(self, checkpoints):
        if self.has_finished_race:
            # Massive bonus for reaching the race end point + speed bonus
            speed_bonus = max(0.0, (2500 - self.total_frames_alive) * 3.0)
            self.fitness = 20000.0 + (self.checkpoints_passed * 1000.0) + speed_bonus
            return self.fitness

        target = (self.current_checkpoint_idx + 1) % len(checkpoints)
        cp = checkpoints[target]
        cp_mid = ((cp[0][0] + cp[1][0]) / 2.0, (cp[0][1] + cp[1][1]) / 2.0)
        dist_to_next = distance((self.x, self.y), cp_mid)
        progress = max(0.0, (250.0 - dist_to_next) / 250.0) * 100.0
        self.fitness = (self.checkpoints_passed * 1000.0) + progress + self.total_distance * 0.02
        return self.fitness

    # ── Drawing ─────────────────────────────────────────────────────────────
    def draw(self, surface, draw_sensors=False):
        if not self.is_alive and not self.is_best and not self.has_finished_race:
            return

        corners = self.get_corners()

        # Sensor rays
        if draw_sensors and self.is_alive and not self.has_finished_race:
            for pt in self.sensor_intersections:
                if pt is not None:
                    pygame.draw.line(
                        surface, (0, 200, 255),
                        (int(self.x), int(self.y)),
                        (int(pt[0]), int(pt[1])), 1
                    )
                    pygame.draw.circle(
                        surface, (255, 60, 90),
                        (int(pt[0]), int(pt[1])), 3
                    )

        # Body color
        if self.has_finished_race:
            body = (255, 215, 0)         # Champion gold
            outline = (255, 255, 255)
        elif self.is_best:
            body = (0, 240, 255)         # Neon cyan leader
            outline = (255, 255, 255)
        elif self.is_alive:
            body = (200, 80, 90)         # Subtle coral
            outline = (240, 140, 140)
        else:
            body = (70, 70, 80)          # Crashed gray
            outline = (90, 90, 100)

        int_corners = [(int(c[0]), int(c[1])) for c in corners]
        pygame.draw.polygon(surface, body, int_corners)
        pygame.draw.polygon(surface, outline, int_corners, 1)

        # Headlight
        rad = math.radians(self.angle)
        fx = self.x + math.cos(rad) * (self.length / 2.0 - 2)
        fy = self.y + math.sin(rad) * (self.length / 2.0 - 2)
        pygame.draw.circle(surface, (255, 255, 200), (int(fx), int(fy)), 3)
