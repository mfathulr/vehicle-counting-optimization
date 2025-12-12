"""Fuzzy logic controller for traffic light optimization."""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from ..config import (
    FUZZY_DURATION_LONG,
    FUZZY_DURATION_MEDIUM,
    FUZZY_DURATION_RANGE,
    FUZZY_DURATION_SHORT,
    FUZZY_MOBIL_HIGH,
    FUZZY_MOBIL_LOW,
    FUZZY_MOBIL_MEDIUM,
    FUZZY_MOBIL_RANGE,
    FUZZY_MOTOR_HIGH,
    FUZZY_MOTOR_LOW,
    FUZZY_MOTOR_MEDIUM,
    FUZZY_MOTOR_RANGE,
)


class TrafficLightOptimizer:
    """Fuzzy logic controller for optimizing traffic light duration."""

    def __init__(self):
        """Initialize the fuzzy logic controller with membership functions and rules."""
        # Define input variables
        self.jumlah_motor = ctrl.Antecedent(
            np.arange(*FUZZY_MOTOR_RANGE), "jumlah_motor"
        )
        self.jumlah_mobil = ctrl.Antecedent(
            np.arange(*FUZZY_MOBIL_RANGE), "jumlah_mobil"
        )

        # Define output variable
        self.duration = ctrl.Consequent(np.arange(*FUZZY_DURATION_RANGE), "duration")

        # Define membership functions for motorcycles
        self.jumlah_motor["low"] = fuzz.trimf(
            self.jumlah_motor.universe, FUZZY_MOTOR_LOW
        )
        self.jumlah_motor["medium"] = fuzz.trimf(
            self.jumlah_motor.universe, FUZZY_MOTOR_MEDIUM
        )
        self.jumlah_motor["high"] = fuzz.trimf(
            self.jumlah_motor.universe, FUZZY_MOTOR_HIGH
        )

        # Define membership functions for cars
        self.jumlah_mobil["low"] = fuzz.trimf(
            self.jumlah_mobil.universe, FUZZY_MOBIL_LOW
        )
        self.jumlah_mobil["medium"] = fuzz.trimf(
            self.jumlah_mobil.universe, FUZZY_MOBIL_MEDIUM
        )
        self.jumlah_mobil["high"] = fuzz.trimf(
            self.jumlah_mobil.universe, FUZZY_MOBIL_HIGH
        )

        # Define membership functions for duration
        self.duration["short"] = fuzz.trimf(
            self.duration.universe, FUZZY_DURATION_SHORT
        )
        self.duration["medium"] = fuzz.trapmf(
            self.duration.universe, FUZZY_DURATION_MEDIUM
        )
        self.duration["long"] = fuzz.trimf(self.duration.universe, FUZZY_DURATION_LONG)

        # Define fuzzy rules
        self._rules = self._create_rules()

        # Create control system
        self.traffic_ctrl = ctrl.ControlSystem(self._rules)
        self.traffic_sim = ctrl.ControlSystemSimulation(self.traffic_ctrl)

    def _create_rules(self):
        """Create fuzzy logic rules for traffic light optimization."""
        rules = [
            # Single input rules
            ctrl.Rule(self.jumlah_motor["low"], self.duration["short"]),
            ctrl.Rule(self.jumlah_motor["medium"], self.duration["medium"]),
            ctrl.Rule(self.jumlah_motor["high"], self.duration["long"]),
            ctrl.Rule(self.jumlah_mobil["low"], self.duration["short"]),
            ctrl.Rule(self.jumlah_mobil["medium"], self.duration["medium"]),
            ctrl.Rule(self.jumlah_mobil["high"], self.duration["long"]),
            # Combined rules
            ctrl.Rule(
                self.jumlah_motor["low"] & self.jumlah_mobil["low"],
                self.duration["short"],
            ),
            ctrl.Rule(
                self.jumlah_motor["low"] & self.jumlah_mobil["medium"],
                self.duration["medium"],
            ),
            ctrl.Rule(
                self.jumlah_motor["low"] & self.jumlah_mobil["high"],
                self.duration["long"],
            ),
            ctrl.Rule(
                self.jumlah_motor["medium"] & self.jumlah_mobil["low"],
                self.duration["medium"],
            ),
            ctrl.Rule(
                self.jumlah_motor["medium"] & self.jumlah_mobil["medium"],
                self.duration["medium"],
            ),
            ctrl.Rule(
                self.jumlah_motor["medium"] & self.jumlah_mobil["high"],
                self.duration["long"],
            ),
            ctrl.Rule(
                self.jumlah_motor["high"] & self.jumlah_mobil["low"],
                self.duration["long"],
            ),
            ctrl.Rule(
                self.jumlah_motor["high"] & self.jumlah_mobil["medium"],
                self.duration["long"],
            ),
            ctrl.Rule(
                self.jumlah_motor["high"] & self.jumlah_mobil["high"],
                self.duration["long"],
            ),
        ]
        return rules

    def optimize(self, num_cars: int, num_motorcycles: int) -> float:
        """
        Calculate optimal traffic light duration based on vehicle counts.

        Args:
            num_cars: Number of cars detected.
            num_motorcycles: Number of motorcycles detected.

        Returns:
            Optimized traffic light duration in seconds (rounded to 2 decimals).
        """
        self.traffic_sim.input["jumlah_mobil"] = num_cars
        self.traffic_sim.input["jumlah_motor"] = num_motorcycles

        # Compute the output
        self.traffic_sim.compute()

        # Return rounded duration
        return np.round(self.traffic_sim.output["duration"], 2)
