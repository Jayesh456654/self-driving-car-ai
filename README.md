# 🏎️ Self-Driving Car AI — MotoGP Championship Edition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pygame 2.5+](https://img.shields.io/badge/Pygame-2.5%2B-green.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero ML Frameworks](https://img.shields.io/badge/Frameworks-Zero%20(Pure%20Math)-orange.svg)]()

An autonomous self-driving car simulation built from scratch in **Python & Pygame**, inspired by [commonLuke's](https://youtu.be/GGWHjAyKJCA) viral AI video *"Can I Make a Better AI Than AI?"*.

In this project, a population of **50 autonomous cars** learns to navigate **9 authentic MotoGP World Championship racetracks** without any pre-recorded driving datasets or high-level machine learning libraries (no PyTorch, no TensorFlow). Instead, the cars evolve their own driving reflexes using **Neuroevolution** — combining **Feedforward Neural Networks** with **Genetic Algorithms (Natural Selection)**.

---

## 📸 Overview & Key Features

* **Pure Mathematical AI:** Custom feedforward neural network and genetic algorithm hand-coded using only Python's standard `math` and `random` libraries.
* **9 Authentic MotoGP Circuits:** Race around procedural recreations of Mugello, Red Bull Ring, Silverstone, Assen, Sachsenring, Phillip Island, COTA, Lusail, and Sepang across 4 difficulty tiers.
* **Transfer Learning:** Switch to any racetrack at any time using number keys `1`–`9` — the cars **preserve their generation count and all learned neural weights**, immediately applying their instincts to the new circuit.
* **Race End Point Auto-Evolution:** When any car successfully completes all checkpoints and crosses the finish line, it receives a massive championship bonus and instantly triggers the next generation seeded by that winner's brain.
* **Real-time Neural Network Visualizer:** Watch the leader's actual brain neurons and synpatic weights fire live in real time as it negotiates corners.
* **Generational Fitness Curve:** In-game telemetry chart tracking the population's intelligence growing generation-by-generation.
* **Virtual LIDAR Sensors:** 5 raycasting distance sensors detecting barriers at `-60°`, `-30°`, `0°`, `+30°`, and `+60°` relative to the car heading, plus current speed.
* **Manual Driving Mode:** Take control of a car yourself at any time using `WASD` or Arrow keys to test your skills against the AI.
* **Simulation Speed Multiplier:** Train at `1x`, `2x`, `5x`, or `10x` fast-forward speed for rapid evolution.
* **Brain Persistence:** Save your champion brain to `best_car_brain.json` and reload it into the fleet on any circuit.

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* Pygame:
  ```bash
  pip install pygame
  ```

### 2. Run the Simulation

**Option A — One-Click (Windows):**
Simply double-click:
```
run.bat
```

**Option B — Terminal:**
```bash
python main.py
```

---

## 🧠 How the AI Works: The Science & Math

This project uses **Neuroevolution**, which replaces traditional gradient descent (backpropagation) with Darwinian evolutionary principles.

```
                           +-------------------------------------+
                           |      50 Cars Spawn (Gen 1)          |
                           |   (Random weights and biases)       |
                           +------------------+------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |         Physics & Sensors           |
                           |   (5 Raycasts measure wall dist)    |
                           +------------------+------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |      Neural Network Forward Pass    |
                           |  Inputs -> Hidden -> Steer/Throttle |
                           +------------------+------------------+
                                              |
                                              v
                      +-----------------------+-----------------------+
                      |                                               |
             [Hits Finish Line]                               [Crashes / Stalls]
                      |                                               |
                      v                                               v
         +--------------------------+                    +--------------------------+
         | Race End Point Triggered |                    | Watchdog kills slow cars |
         | +20,000 Fitness Bonus    |                    | Evaluates checkpoint fit |
         +------------+-------------+                    +------------+-------------+
                      |                                               |
                      +-----------------------+-----------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |          Natural Selection          |
                           |  - Top 4 Elites kept 100% untouched |
                           |  - Top 30% selected as parents      |
                           |  - 50/50 Crossover & 9% Mutation    |
                           +------------------+------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |     Spawn Generation N + 1          |
                           +-------------------------------------+
```

### 1. Perception: LIDAR Raycasting
Each car casts 5 virtual distance sensor rays at angles:
$$\theta_{\text{ray}} \in \{-60^\circ, -30^\circ, 0^\circ, +30^\circ, +60^\circ\}$$

Using line-segment intersection between ray segment $p_1 \to p_2$ and track wall segment $p_3 \to p_4$:

$$t = \frac{(x_1 - x_3)(y_3 - y_4) - (y_1 - y_3)(x_3 - x_4)}{(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4)}$$

$$u = \frac{(x_1 - x_3)(y_1 - y_2) - (y_1 - y_3)(x_1 - x_2)}{(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4)}$$

If $0 \le t \le 1$ and $0 \le u \le 1$, an intersection occurs at:
$$(x, y) = (x_1 + t(x_2 - x_1), \; y_1 + t(y_2 - y_1))$$

Distance is normalized to $[0.0, 1.0]$ ($0.0 = \text{touching wall}, 1.0 = \text{open road}$).

### 2. The Brain: Feedforward Network (`[6, 10, 4, 2]`)
* **Input Layer (6 neurons):** 5 normalized ray distances + 1 normalized speed ($v / v_{\max}$).
* **Hidden Layer 1 (10 neurons):** Feature extraction with $\tanh$ activation.
* **Hidden Layer 2 (4 neurons):** Decision refinement with $\tanh$ activation.
* **Output Layer (2 neurons):**
  * **Output 0 (Steering):** Value $\in [-1.0, 1.0]$ mapped to $\Delta \theta = \text{output}_0 \times \text{turn\_rate}$.
  * **Output 1 (Throttle/Brake):** Value $\in [-1.0, 1.0]$ where $>0$ accelerates and $<0$ brakes.

$$\mathbf{a}^{(l)} = \tanh\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

### 3. The Fitness Function
Fitness measures how well a car navigates the circuit:

$$\text{Fitness} = (\text{Checkpoints Passed} \times 1000) + \text{Progress to Next Checkpoint} + (\text{Distance Traveled} \times 0.02)$$

When a car reaches the **Race End Point (Finish Line)**:
$$\text{Fitness}_{\text{winner}} = 20000 + (\text{Checkpoints Passed} \times 1000) + \max(0, (2500 - \text{Frames Alive}) \times 3)$$

### 4. Genetic Evolution Mechanics
* **Elitism (Top 4):** The 4 highest-scoring cars survive completely unmodified into the next generation. Progress is never lost!
* **Parent Selection:** Mating pool formed from the top 30% of performers.
* **Uniform Crossover:** Offspring randomly inherit weights and biases 50/50 from two parents.
* **Gaussian Mutation:** 9% probability per weight/bias of adding random Gaussian noise $\mathcal{N}(0, 0.28)$, introducing new driving reflexes without destroying existing knowledge.
* **Inactivity Watchdog:** Cars that get stuck, drive backwards, or stall are eliminated after a short timeout so generations don't waste time.

---

## 🏁 The 9 MotoGP Circuits

The track generator uses **Catmull-Rom cubic splines** to construct realistic continuous boundary curves and dynamic track widths:

| # | Circuit | Country | Difficulty | Track Width | Corner Highlights |
|:---:|:---|:---|:---:|:---:|:---|
| **`1`** | **Mugello** | Italy 🇮🇹 | `Medium` | 62 px | Massive 1.1km Tuscan straight, San Donato hairpin & Arrabbiata sweeps |
| **`2`** | **Red Bull Ring** | Austria 🇦🇹 | `Very Hard` | 58 px | Steep uphill climb into sharp Remus hairpin & downhill chicane |
| **`3`** | **Silverstone** | United Kingdom 🇬🇧 | `Hard` | 64 px | High-speed Maggots, Becketts & Chapel into Wellington straight |
| **`4`** | **Custom Circuit** | Custom 🛠️ | `Extreme` | 52 px | Custom user-drawn Grand Prix: crest straight, technical double-hooks & infield loops |
| **`5`** | **Sachsenring** | Germany 🇩🇪 | `Extreme` | 56 px | Technical mountain rollercoaster, continuous Omega carousel & Waterfall |
| **`6`** | **Phillip Island** | Australia 🇦🇺 | `Hard` | 64 px | Seaside high-speed sweepers: Doohan Corner & Stoner sweepers |
| **`7`** | **COTA (Austin)** | United States 🇺🇸 | `Extreme` | 58 px | 41m uphill climb into blind Turn 1 hairpin, rapid snake esses |
| **`8`** | **Lusail Circuit** | Qatar 🇶🇦 | `Medium` | 66 px | 1.06km desert main straight into sweeping night corners |
| **`9`** | **Sepang Circuit** | Malaysia 🇲🇾 | `Hard` | 62 px | Twin parallel straights, sweeping curves & heavy braking hairpins |

---

## ⌨️ Controls & Keybindings

| Key | Action |
|:---|:---|
| **`1` – `9`** | **Quick Jump** directly to any of the 9 MotoGP circuits *(Preserves generation & trained brains)* |
| **`T`** / **`Y`** | Cycle to Next / Previous MotoGP circuit *(Preserves generation & trained brains)* |
| **`F`** | Cycle simulation speed multiplier: `1x` $\to$ `2x` $\to$ `5x` $\to$ `10x` |
| **`SPACE`** | Force natural selection & spawn next generation immediately |
| **`M`** | Toggle **Manual Driving Mode** (Drive with `WASD` or Arrow keys) |
| **`R`** | Cycle sensor rays display (`Leader only` $\to$ `All cars` $\to$ `Off`) |
| **`C`** | Toggle checkpoint gate lines visibility |
| **`N`** | Reset population to Generation 1 (Fresh random brains) |
| **`S`** | Save current best neural network brain to `best_car_brain.json` |
| **`L`** | Load saved brain into the entire fleet |
| **`ESC`** | Quit simulation |

---

## 📁 Repository Structure

```
self_driving_car_ai/
├── main.py              # Main Pygame application, HUD, live NN visualizer, fitness graph
├── car.py               # Car physics, raycasting sensors, collisions, fitness, race end point
├── neural_network.py    # Pure Python feedforward neural network (forward, mutate, crossover, save/load)
├── population.py        # Genetic Algorithm manager (selection, elitism, breeding, transfer learning)
├── track.py             # 9 MotoGP circuits with Catmull-Rom splines, curbs, checkpoints, finish banner
├── geometry.py          # Vector math & high-speed line segment intersection algorithms
├── best_car_brain.json  # Pre-trained champion neural network brain
├── run.bat              # One-click Windows batch launcher
├── requirements.txt     # Python dependencies (pygame)
├── .gitignore           # Git ignore file
└── README.md            # Comprehensive project documentation
```

---

## 💡 Training & Experimentation Tips

1. **Bootstrapping on Flowing Tracks:** Start training on **Mugello (`Key 1`)** or **Lusail (`Key 8`)** at `10x` speed (`F` key). Because these tracks are wider (62–66px), cars quickly discover forward acceleration and basic wall avoidance within 5–8 generations.
2. **Transfer Learning to Extreme Tracks:** Once cars can lap Mugello, press **`5`** (Sachsenring) or **`7`** (COTA). The cars immediately carry over their learned steering reflexes and begin tackling tight hairpins!
3. **Saving Checkpoints:** Press **`S`** whenever a car completes an impressive lap. You can reload it anytime with **`L`**.
4. **Human Challenge:** Press **`M`** to switch to Manual mode and see if you can beat the AI leader's lap time!

---

## 📜 License & Credits

* Inspired by **commonLuke's** YouTube video *"Can I Make a Better AI Than AI?"*.
* Released under the **MIT License**. Feel free to use, modify, and distribute for personal and educational projects!
