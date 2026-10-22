"""
track.py - MotoGP World Championship Circuits with Catmull-Rom splines,
boundary calculations, checkpoints, dynamic width, and authentic circuit styling.
Includes clear visual rendering of the Race End Point (Finish Line).
"""
import math
import pygame
from geometry import line_intersection


def interpolate_loop(pts, steps_per_segment=4):
    """Smooths waypoints using Catmull-Rom spline interpolation (closed loop)."""
    n = len(pts)
    smooth = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for s in range(steps_per_segment):
            t = s / steps_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            smooth.append((x, y))
    return smooth


# ── Official MotoGP World Championship Circuits ─────────────────────────────
MOTOGP_CIRCUITS = {
    "Mugello (Italy)": {
        "width": 62,
        "difficulty": "Medium",
        "country": "Italy 🇮🇹",
        "description": "San Donato hairpin, Casanova-Savelli & legendary Arrabbiata sweeps",
        "waypoints": [
            (250, 670), (500, 670), (750, 670), (950, 670),
            (1100, 620), (1170, 500), (1140, 360), (1050, 240),
            (920, 160), (760, 200), (620, 140), (480, 160),
            (360, 230), (250, 340), (180, 460), (190, 580)
        ]
    },
    "Red Bull Ring (Austria)": {
        "width": 58,
        "difficulty": "Very Hard",
        "country": "Austria 🇦🇹",
        "description": "Steep uphill power straight into the sharp Remus hairpin & downhill chicane",
        "waypoints": [
            (280, 670), (540, 670), (800, 670), (1020, 660),
            (1150, 590), (1170, 450), (1110, 310), (990, 190),
            (840, 130), (670, 120), (500, 140), (360, 190),
            (250, 270), (190, 380), (180, 500), (200, 600)
        ]
    },
    "Silverstone (UK)": {
        "width": 64,
        "difficulty": "Hard",
        "country": "United Kingdom 🇬🇧",
        "description": "Ultra-fast flowing Maggots, Becketts & Chapel into Wellington Straight",
        "waypoints": [
            (250, 670), (500, 670), (750, 670), (960, 670),
            (1110, 610), (1170, 480), (1130, 340), (1030, 220),
            (880, 150), (720, 130), (550, 150), (400, 180),
            (280, 250), (200, 360), (170, 480), (180, 590)
        ]
    },
    "Custom Circuit (User Track)": {
        "width": 52,
        "difficulty": "Extreme",
        "country": "Custom 🛠️",
        "description": "Custom Grand Prix: long crest straight into technical double-hooks and infield loops",
        "waypoints": [
            (152.6, 150.1),
            (264.3, 231.1),
            (382.3, 199.9),
            (502.1, 130.4),
            (634.6, 108.0),
            (769.1, 92.7),
            (905.7, 85.0),
            (1038.1, 104.2),
            (1139.0, 221.5),
            (1180.0, 407.7),
            (1124.9, 584.9),
            (1012.8, 680.0),
            (943.9, 567.0),
            (1017.3, 424.2),
            (922.5, 346.9),
            (868.4, 524.8),
            (795.8, 586.8),
            (723.2, 538.1),
            (615.7, 632.8),
            (498.6, 583.9),
            (521.0, 475.7),
            (645.0, 410.0),
            (630.0, 220.0),
            (540.0, 250.0),
            (510.0, 380.0),
            (402.7, 355.1),
            (303.3, 486.5),
            (294.7, 639.4),
            (177.6, 653.5),
            (105.0, 537.0),
            (155.4, 447.6),
            (80.0, 288.8),
        ]
    },
    "Sachsenring (Germany)": {
        "width": 56,
        "difficulty": "Extreme",
        "country": "Germany 🇩🇪",
        "description": "Ultra-tight mountain roller-coaster, continuous Omega carousel & Waterfall plunge",
        "waypoints": [
            (260, 670), (520, 670), (780, 670), (980, 660),
            (1120, 600), (1170, 470), (1120, 330), (1010, 210),
            (870, 140), (710, 130), (550, 140), (410, 180),
            (290, 250), (210, 350), (170, 470), (180, 590)
        ]
    },
    "Phillip Island (Australia)": {
        "width": 64,
        "difficulty": "Hard",
        "country": "Australia 🇦🇺",
        "description": "High-speed coastal ocean circuit: Doohan Corner, Southern Loop & Stoner sweepers",
        "waypoints": [
            (250, 670), (500, 670), (760, 670), (980, 660),
            (1130, 600), (1170, 460), (1120, 320), (1020, 200),
            (880, 140), (730, 130), (570, 140), (430, 180),
            (310, 250), (220, 360), (180, 480), (190, 590)
        ]
    },
    "Circuit of The Americas (USA)": {
        "width": 58,
        "difficulty": "Extreme",
        "country": "United States 🇺🇸",
        "description": "Steep 41m uphill blind Turn 1 hairpin, followed by rapid-fire snake esses",
        "waypoints": [
            (270, 670), (530, 670), (790, 670), (1010, 660),
            (1140, 600), (1170, 470), (1130, 330), (1030, 210),
            (900, 150), (750, 190), (610, 130), (470, 170),
            (350, 240), (250, 340), (190, 460), (200, 590)
        ]
    },
    "Lusail Circuit (Qatar)": {
        "width": 66,
        "difficulty": "Medium",
        "country": "Qatar 🇶🇦",
        "description": "Massive 1.06km desert straight into sweeping medium-speed night corners",
        "waypoints": [
            (240, 670), (500, 670), (760, 670), (990, 660),
            (1130, 590), (1170, 450), (1120, 310), (1020, 190),
            (880, 140), (720, 130), (560, 140), (410, 180),
            (290, 250), (210, 360), (180, 480), (180, 590)
        ]
    },
    "Sepang Circuit (Malaysia)": {
        "width": 62,
        "difficulty": "Hard",
        "country": "Malaysia 🇲🇾",
        "description": "Tactical Grand Prix layout with twin parallel straights & tight hairpin complex",
        "waypoints": [
            (260, 670), (520, 670), (780, 670), (1000, 660),
            (1140, 600), (1170, 460), (1120, 320), (1020, 200),
            (880, 140), (730, 170), (600, 130), (460, 160),
            (340, 230), (240, 330), (180, 450), (190, 580)
        ]
    }
}

TRACK_PRESETS = {k: v["waypoints"] for k, v in MOTOGP_CIRCUITS.items()}


class Track:
    def __init__(self, name="Mugello (Italy)", track_width=None):
        if name not in MOTOGP_CIRCUITS:
            name = list(MOTOGP_CIRCUITS.keys())[0]

        circuit_info = MOTOGP_CIRCUITS[name]
        self.name = name
        self.country = circuit_info.get("country", "")
        self.difficulty = circuit_info.get("difficulty", "Medium")
        self.description = circuit_info.get("description", "")
        
        self.track_width = track_width if track_width is not None else circuit_info.get("width", 62)
        waypoints = circuit_info["waypoints"]

        self.center_points = interpolate_loop(waypoints, steps_per_segment=4)
        self.n = len(self.center_points)

        self.inner_poly = []
        self.outer_poly = []
        self.inner_walls = []
        self.outer_walls = []
        self.checkpoints = []

        self._build_boundaries()

        # Start configuration
        p0 = self.center_points[0]
        p1 = self.center_points[1]
        self.start_x = p0[0]
        self.start_y = p0[1]
        self.start_angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    def _build_boundaries(self):
        half_w = self.track_width / 2.0
        n = self.n

        for i in range(n):
            prev_pt = self.center_points[(i - 1) % n]
            curr_pt = self.center_points[i]
            next_pt = self.center_points[(i + 1) % n]

            tx = next_pt[0] - prev_pt[0]
            ty = next_pt[1] - prev_pt[1]
            t_len = math.hypot(tx, ty)
            if t_len == 0:
                tx, ty = 1, 0
            else:
                tx /= t_len
                ty /= t_len

            nx = -ty
            ny = tx

            ix = curr_pt[0] - nx * half_w
            iy = curr_pt[1] - ny * half_w
            ox = curr_pt[0] + nx * half_w
            oy = curr_pt[1] + ny * half_w

            self.inner_poly.append((ix, iy))
            self.outer_poly.append((ox, oy))
            self.checkpoints.append(((ix, iy), (ox, oy)))

        for i in range(n):
            self.inner_walls.append(
                (self.inner_poly[i], self.inner_poly[(i + 1) % n])
            )
            self.outer_walls.append(
                (self.outer_poly[i], self.outer_poly[(i + 1) % n])
            )

    def get_all_walls(self):
        return self.inner_walls + self.outer_walls

    # ── Drawing ─────────────────────────────────────────────────────────────
    def draw(self, surface, show_checkpoints=False, font=None):
        """Draws asphalt road, racing curbs, barriers, and the Race End Point finish line."""
        asphalt = (38, 42, 50)
        curb_red = (225, 45, 60)
        curb_white = (245, 245, 250)
        border = (75, 90, 105)

        # Road surface
        for i in range(self.n):
            p1 = self.inner_poly[i]
            p2 = self.outer_poly[i]
            p3 = self.outer_poly[(i + 1) % self.n]
            p4 = self.inner_poly[(i + 1) % self.n]
            pygame.draw.polygon(surface, asphalt, [p1, p2, p3, p4])

        # Racing Curbs
        for i in range(self.n):
            c = curb_red if (i % 2 == 0) else curb_white
            pygame.draw.line(surface, c,
                             self.outer_poly[i],
                             self.outer_poly[(i + 1) % self.n], 4)
            pygame.draw.line(surface, c,
                             self.inner_poly[i],
                             self.inner_poly[(i + 1) % self.n], 4)

        # Barriers
        for w in self.inner_walls:
            pygame.draw.line(surface, border, w[0], w[1], 1)
        for w in self.outer_walls:
            pygame.draw.line(surface, border, w[0], w[1], 1)

        # Racing line guide dashes
        for i in range(0, self.n, 3):
            p_a = self.center_points[i]
            p_b = self.center_points[(i + 1) % self.n]
            pygame.draw.line(surface, (50, 56, 68), p_a, p_b, 1)

        # Checkpoints
        if show_checkpoints:
            for cp in self.checkpoints:
                pygame.draw.line(surface, (45, 65, 85), cp[0], cp[1], 1)

        # ── RACE END POINT (START / FINISH LINE) ──────────────────────────
        scp = self.checkpoints[0]
        p_in = scp[0]
        p_out = scp[1]

        # Draw glowing finish line bar
        pygame.draw.line(surface, (255, 215, 0), p_in, p_out, 6)

        # Draw checkered pattern along the finish line
        num_blocks = 8
        for b in range(num_blocks):
            t1 = b / num_blocks
            t2 = (b + 1) / num_blocks
            x1 = p_in[0] + (p_out[0] - p_in[0]) * t1
            y1 = p_in[1] + (p_out[1] - p_in[1]) * t1
            x2 = p_in[0] + (p_out[0] - p_in[0]) * t2
            y2 = p_in[1] + (p_out[1] - p_in[1]) * t2
            c = (255, 255, 255) if (b % 2 == 0) else (20, 20, 25)
            pygame.draw.line(surface, c, (x1, y1), (x2, y2), 4)

        # Label for RACE END POINT
        mx = (p_in[0] + p_out[0]) / 2.0
        my = (p_in[1] + p_out[1]) / 2.0

        if font is not None:
            lbl = font.render("🏁 RACE END POINT", True, (255, 215, 0))
            surface.blit(lbl, (mx - lbl.get_width() // 2, my + 10))
        else:
            pygame.draw.circle(surface, (255, 215, 0), (int(mx), int(my)), 5)
