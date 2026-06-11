"""
ai_grand_prix_controller.py

Drop-in direction + speed controller for an AI Grand Prix style car.

Main function:
    calculate_controls(sensor_distances)

Input:
    sensor_distances:
        A list/tuple/numpy array of distance readings from the car's sensors.
        Assumption: readings are ordered from LEFT to RIGHT across the front of the car.
        Example with 7 sensors:
            [-90°, -60°, -30°, 0°, +30°, +60°, +90°]

Output:
    steering, speed, debug

    steering:
        Float between -1.0 and +1.0
        -1.0 = hard left
         0.0 = straight
        +1.0 = hard right

    speed:
        Float between MIN_SPEED and MAX_SPEED

    debug:
        Dictionary explaining the decision. Useful for UI/accessibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Any

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "This controller requires numpy. Install it with: pip install numpy"
    ) from exc


# -----------------------------
# Tune these values for your game
# -----------------------------

MIN_SPEED = 0.35
MAX_SPEED = 3.00

# Distance at which the car considers the path ahead very safe.
CLEAR_FRONT_DISTANCE = 6.0

# Any sensor reading below this is treated as dangerous.
DANGER_DISTANCE = 0.75

# Distance needed for a sensor direction to count as safe.
SAFE_DISTANCE = 1.10

# Maximum physical steering angle represented by steering = +/- 1.
MAX_STEERING_ANGLE_DEGREES = 45

# Higher = smoother but slower to react.
STEERING_SMOOTHING = 0.28

# LiDAR smoothing window. Must be odd-ish; 5 or 7 is usually good.
SENSOR_SMOOTHING_WINDOW = 5

# Safety bubble around close obstacles, as a fraction of the sensor count.
# Larger values make the car more cautious.
SAFETY_BUBBLE_FRACTION = 0.08

# Prefer driving forward over aiming at extreme left/right side gaps.
# Higher = stronger preference for straight driving.
FORWARD_BIAS_POWER = 1.35

# How much to slow down while turning.
CORNER_SLOWDOWN_STRENGTH = 0.62

# Straight-line boost when the front is clear and steering is small.
STRAIGHT_BOOST = 1.18


@dataclass
class ControllerState:
    """
    Stores previous control values so the car moves smoothly over time.
    Keep one instance of this for your car.
    """
    previous_steering: float = 0.0
    previous_speed: float = MIN_SPEED


def clamp(value: float, low: float, high: float) -> float:
    """Restrict value to the range [low, high]."""
    return max(low, min(high, value))


def clean_sensor_data(sensor_distances: Iterable[float], max_distance: float = 10.0) -> np.ndarray:
    """
    Converts raw sensor data into a clean numpy array.
    Handles NaN, infinity, negative values, and empty inputs.
    """
    distances = np.array(list(sensor_distances), dtype=float)

    if distances.size == 0:
        # Emergency fallback: fake one close obstacle straight ahead.
        return np.array([0.1], dtype=float)

    distances = np.nan_to_num(
        distances,
        nan=0.0,
        posinf=max_distance,
        neginf=0.0,
    )

    distances = np.clip(distances, 0.0, max_distance)
    return distances


def smooth_distances(distances: np.ndarray, window: int = SENSOR_SMOOTHING_WINDOW) -> np.ndarray:
    """
    Smooths sensor readings so the car does not twitch because of one noisy ray.
    """
    if distances.size < 3:
        return distances

    window = max(1, min(window, distances.size))

    kernel = np.ones(window) / window
    return np.convolve(distances, kernel, mode="same")


def find_largest_safe_gap(safe_mask: np.ndarray) -> Tuple[int, int]:
    """
    Finds the largest continuous stretch of safe sensor readings.

    Returns:
        start_index, end_index
        end_index is exclusive.
    """
    best_start = 0
    best_end = 0
    current_start = None

    for i, is_safe in enumerate(safe_mask):
        if is_safe and current_start is None:
            current_start = i

        elif not is_safe and current_start is not None:
            if i - current_start > best_end - best_start:
                best_start, best_end = current_start, i
            current_start = None

    if current_start is not None:
        if len(safe_mask) - current_start > best_end - best_start:
            best_start, best_end = current_start, len(safe_mask)

    return best_start, best_end


def choose_target_angle(distances: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Chooses the direction the car should aim for.

    Formula:
        score(angle) = distance(angle) * forward_bias(angle)

    Where:
        forward_bias(angle) = cos(angle) ^ FORWARD_BIAS_POWER

    This means:
        - Big open gaps are attractive.
        - Directions closer to straight ahead are preferred.
        - Extreme side directions are used only when needed.
    """
    n = distances.size
    angles = np.linspace(-math.pi / 2, math.pi / 2, n)

    smoothed = smooth_distances(distances)

    safe = smoothed > SAFE_DISTANCE

    # Add a safety bubble around the closest obstacle.
    closest_index = int(np.argmin(smoothed))
    bubble_size = max(1, int(n * SAFETY_BUBBLE_FRACTION))

    bubble_start = max(0, closest_index - bubble_size)
    bubble_end = min(n, closest_index + bubble_size + 1)
    safe[bubble_start:bubble_end] = False

    gap_start, gap_end = find_largest_safe_gap(safe)

    if gap_end <= gap_start:
        # No safe gap found. Turn away from the closest obstacle.
        closest_angle = angles[closest_index]
        emergency_target_angle = -closest_angle

        debug = {
            "mode": "emergency_turn",
            "closest_index": closest_index,
            "closest_distance": float(smoothed[closest_index]),
            "target_angle_degrees": math.degrees(emergency_target_angle),
            "gap_start": None,
            "gap_end": None,
        }

        return emergency_target_angle, debug

    gap_distances = smoothed[gap_start:gap_end]
    gap_angles = angles[gap_start:gap_end]

    # Prefer forward angles, but still allow side gaps when necessary.
    forward_bias = np.maximum(np.cos(gap_angles), 0.0) ** FORWARD_BIAS_POWER

    # Main direction formula.
    scores = gap_distances * forward_bias

    best_local_index = int(np.argmax(scores))
    best_global_index = gap_start + best_local_index
    target_angle = float(angles[best_global_index])

    debug = {
        "mode": "follow_largest_gap",
        "closest_index": closest_index,
        "closest_distance": float(smoothed[closest_index]),
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "best_index": int(best_global_index),
        "target_angle_degrees": float(math.degrees(target_angle)),
        "best_distance": float(smoothed[best_global_index]),
    }

    return target_angle, debug


def angle_to_steering(target_angle_radians: float) -> float:
    """
    Converts target angle into normalized game steering.

    Formula:
        steering = target_angle / max_steering_angle

    Result:
        -1.0 = full left
         0.0 = straight
        +1.0 = full right
    """
    max_angle = math.radians(MAX_STEERING_ANGLE_DEGREES)
    return clamp(target_angle_radians / max_angle, -1.0, 1.0)


def calculate_speed(
    distances: np.ndarray,
    steering: float,
    target_angle_radians: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculates speed from front safety and steering angle.

    Formula:
        front_safety = front_distance / CLEAR_FRONT_DISTANCE
        corner_factor = 1 - abs(steering) * CORNER_SLOWDOWN_STRENGTH

        speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * front_safety * corner_factor

    Extra logic:
        - Emergency braking if very close to an obstacle.
        - Boost on clear straights.
    """
    n = distances.size
    center = n // 2

    smoothed = smooth_distances(distances)

    # Average the central front rays rather than relying on one ray.
    front_radius = max(1, n // 12)
    front_start = max(0, center - front_radius)
    front_end = min(n, center + front_radius + 1)

    front_distance = float(np.mean(smoothed[front_start:front_end]))
    closest_distance = float(np.min(smoothed))

    front_safety = clamp(front_distance / CLEAR_FRONT_DISTANCE, 0.0, 1.0)

    corner_factor = 1.0 - abs(steering) * CORNER_SLOWDOWN_STRENGTH
    corner_factor = clamp(corner_factor, 0.30, 1.0)

    speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * front_safety * corner_factor

    # Fast on safe straights.
    if abs(steering) < 0.14 and front_distance > CLEAR_FRONT_DISTANCE * 0.75:
        speed *= STRAIGHT_BOOST

    # Slow if anything is dangerously close.
    if closest_distance < DANGER_DISTANCE:
        danger_factor = clamp(closest_distance / DANGER_DISTANCE, 0.25, 1.0)
        speed *= danger_factor

    speed = clamp(speed, MIN_SPEED, MAX_SPEED)

    debug = {
        "front_distance": front_distance,
        "closest_distance": closest_distance,
        "front_safety": front_safety,
        "corner_factor": corner_factor,
        "speed_before_clamp": speed,
    }

    return speed, debug


def smooth_controls(
    steering: float,
    speed: float,
    state: ControllerState,
) -> Tuple[float, float]:
    """
    Smooths steering and speed for nicer game mechanics.

    Formula:
        smoothed = alpha * new + (1 - alpha) * previous
    """
    alpha = STEERING_SMOOTHING

    smooth_steering = alpha * steering + (1.0 - alpha) * state.previous_steering

    # Speed can react a little faster than steering.
    speed_alpha = 0.42
    smooth_speed = speed_alpha * speed + (1.0 - speed_alpha) * state.previous_speed

    state.previous_steering = smooth_steering
    state.previous_speed = smooth_speed

    return smooth_steering, smooth_speed


def calculate_controls(
    sensor_distances: Iterable[float],
    state: ControllerState | None = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Main function to call from your game.

    Example:
        state = ControllerState()

        while game_running:
            lidar = get_lidar_from_game()
            steering, speed, debug = calculate_controls(lidar, state)
            car.set_controls(steering=steering, speed=speed)

    Returns:
        steering, speed, debug
    """
    if state is None:
        state = ControllerState()

    distances = clean_sensor_data(sensor_distances)

    target_angle, direction_debug = choose_target_angle(distances)
    raw_steering = angle_to_steering(target_angle)

    raw_speed, speed_debug = calculate_speed(
        distances=distances,
        steering=raw_steering,
        target_angle_radians=target_angle,
    )

    steering, speed = smooth_controls(raw_steering, raw_speed, state)

    risk = "SAFE"
    if speed_debug["closest_distance"] < DANGER_DISTANCE:
        risk = "DANGER"
    elif abs(steering) > 0.55 or speed_debug["front_distance"] < 2.0:
        risk = "CAUTION"

    debug = {
        "steering": float(steering),
        "speed": float(speed),
        "raw_steering": float(raw_steering),
        "raw_speed": float(raw_speed),
        "risk": risk,
        "decision": explain_decision(steering, speed, direction_debug, speed_debug),
        "direction": direction_debug,
        "speed_logic": speed_debug,
    }

    return float(steering), float(speed), debug


def explain_decision(
    steering: float,
    speed: float,
    direction_debug: Dict[str, Any],
    speed_debug: Dict[str, Any],
) -> str:
    """
    Human-readable decision explanation for an accessibility UI.
    """
    closest = speed_debug["closest_distance"]
    front = speed_debug["front_distance"]

    if closest < DANGER_DISTANCE:
        return "Danger: obstacle very close. Slowing down and turning toward the safest gap."

    if direction_debug["mode"] == "emergency_turn":
        return "No clear gap found. Emergency turn away from closest obstacle."

    if steering > 0.25:
        turn_text = "steering right"
    elif steering < -0.25:
        turn_text = "steering left"
    else:
        turn_text = "driving mostly straight"

    if front > CLEAR_FRONT_DISTANCE * 0.75:
        speed_text = "front path is clear, allowing higher speed"
    elif front > 2.0:
        speed_text = "moderate space ahead, using controlled speed"
    else:
        speed_text = "limited space ahead, slowing down"

    return f"Safest gap selected; {turn_text}; {speed_text}."


# -----------------------------
# Optional local test
# -----------------------------

if __name__ == "__main__":
    state = ControllerState()

    # Example: more space slightly to the right.
    fake_lidar = [
        0.7, 0.9, 1.2, 1.8, 2.8, 4.5, 5.5, 5.2, 4.0, 2.0, 1.0, 0.8
    ]

    steering, speed, debug = calculate_controls(fake_lidar, state)

    print("Steering:", round(steering, 3))
    print("Speed:", round(speed, 3))
    print("Risk:", debug["risk"])
    print("Decision:", debug["decision"])
