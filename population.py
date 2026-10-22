"""
population.py - Genetic Algorithm population manager for evolving self-driving cars.
Handles evolution triggers on race end point completion and extinction.
"""
import random
from car import Car


class Population:
    def __init__(self, size=50, start_x=250, start_y=670, start_angle=0.0):
        self.size = size
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        self.generation = 1
        self.cars = [Car(start_x, start_y, start_angle) for _ in range(size)]

        self.best_car = self.cars[0]
        self.best_car.is_best = True
        self.current_best_fitness = 0.0
        self.all_time_best_fitness = 0.0
        self.all_time_best_brain = None
        self.history = []  # best fitness per generation
        self.total_lap_completions = 0

    def update(self, walls, checkpoints):
        """
        Updates all living cars, tracks leader, and checks if any car reached the race end point.
        Returns the winning car if a car hit the race end point, otherwise None.
        """
        winner = None
        for car in self.cars:
            if car.is_alive:
                car.update(walls, checkpoints)
                car.calculate_fitness(checkpoints)
                if car.has_finished_race and winner is None:
                    winner = car

        if self.cars:
            sorted_cars = sorted(self.cars, key=lambda c: c.fitness, reverse=True)
            for c in self.cars:
                c.is_best = False
            self.best_car = sorted_cars[0]
            self.best_car.is_best = True
            self.current_best_fitness = self.best_car.fitness

            if self.current_best_fitness > self.all_time_best_fitness:
                self.all_time_best_fitness = self.current_best_fitness
                self.all_time_best_brain = self.best_car.brain.clone()

        if winner is not None:
            self.total_lap_completions += 1

        return winner

    def is_extinct(self):
        """Returns True if no cars are alive and none finished."""
        return not any(c.is_alive and not c.has_finished_race for c in self.cars)

    def count_alive(self):
        return sum(1 for c in self.cars if c.is_alive and not c.has_finished_race)

    def transfer_to_track(self, start_x, start_y, start_angle):
        """
        Transfers existing trained population to a new track without resetting generation or brains!
        Reposition cars at the new start position and orientation while preserving all learned weights.
        """
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle
        self.current_best_fitness = 0.0

        for car in self.cars:
            car.reset_to_start(start_x, start_y, start_angle)

        if self.cars:
            self.best_car = self.cars[0]
            self.best_car.is_best = True

    def natural_selection(self, checkpoints):
        """Evolves the population using tournament selection, crossover, and elitism."""
        for car in self.cars:
            car.calculate_fitness(checkpoints)

        sorted_cars = sorted(self.cars, key=lambda c: c.fitness, reverse=True)
        self.history.append(sorted_cars[0].fitness)

        if sorted_cars[0].fitness > self.all_time_best_fitness:
            self.all_time_best_fitness = sorted_cars[0].fitness
            self.all_time_best_brain = sorted_cars[0].brain.clone()

        # Mating pool: top 30%
        pool_size = max(3, int(self.size * 0.30))
        pool = sorted_cars[:pool_size]

        new_cars = []

        # Elitism: keep top 4 intact
        elite_n = min(4, self.size)
        for i in range(elite_n):
            new_cars.append(
                Car(self.start_x, self.start_y, self.start_angle,
                    brain=sorted_cars[i].brain.clone())
            )

        # Also add a clone of the all-time best
        if self.all_time_best_brain:
            b = self.all_time_best_brain.clone()
            b.mutate(rate=0.03, magnitude=0.1)
            new_cars.append(
                Car(self.start_x, self.start_y, self.start_angle, brain=b)
            )

        # Fill rest with offspring
        while len(new_cars) < self.size:
            pa = random.choice(pool)
            pb = random.choice(pool)
            if random.random() < 0.65:
                child_brain = pa.brain.crossover(pb.brain)
            else:
                child_brain = pa.brain.clone()
            child_brain.mutate(rate=0.09, magnitude=0.28)
            new_cars.append(
                Car(self.start_x, self.start_y, self.start_angle, brain=child_brain)
            )

        self.cars = new_cars
        self.generation += 1
        self.best_car = self.cars[0]
        self.best_car.is_best = True

    def reset(self):
        """Fresh random population."""
        self.generation = 1
        self.cars = [Car(self.start_x, self.start_y, self.start_angle)
                     for _ in range(self.size)]
        self.best_car = self.cars[0]
        self.best_car.is_best = True
        self.current_best_fitness = 0.0
        self.all_time_best_fitness = 0.0
        self.all_time_best_brain = None
        self.history = []
        self.total_lap_completions = 0

    def save_best_brain(self, filepath="best_car_brain.json"):
        brain = self.all_time_best_brain or (self.best_car.brain if self.best_car else None)
        if brain:
            brain.save(filepath)
            return True
        return False

    def load_brain_for_all(self, filepath="best_car_brain.json"):
        from neural_network import NeuralNetwork
        loaded = NeuralNetwork.load(filepath)
        self.all_time_best_brain = loaded.clone()
        self.all_time_best_fitness = 0.0

        new_cars = []
        for _ in range(5):
            new_cars.append(
                Car(self.start_x, self.start_y, self.start_angle, brain=loaded.clone())
            )
        while len(new_cars) < self.size:
            b = loaded.clone()
            b.mutate(rate=0.06, magnitude=0.18)
            new_cars.append(
                Car(self.start_x, self.start_y, self.start_angle, brain=b)
            )
        self.cars = new_cars
        self.generation = 1
        self.best_car = self.cars[0]
        self.best_car.is_best = True
