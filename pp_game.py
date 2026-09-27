import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import tensorflow as tf

# Load hand gesture model
model = tf.keras.models.load_model("hand_gesture_model.keras")

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Ball settings
ball_radius = 15
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_speed_x = 9 * random.choice((1, -1))  # Increased speed
ball_speed_y = 9 * random.choice((1, -1))

# Paddle settings
paddle_width = 20
paddle_height = 100
left_paddle_x = 50
right_paddle_x = WIDTH - 70
left_paddle_y = HEIGHT // 2 - paddle_height // 2
right_paddle_y = HEIGHT // 2 - paddle_height // 2
paddle_speed = 10
smoothing_factor = 0.2  # Helps reduce fluctuations

# Score variables
left_score = 0
right_score = 0
font = pygame.font.Font(None, 36)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture video
cap = cv2.VideoCapture(0)
clock = pygame.time.Clock()

running = True
while running:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_positions = {}
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = WIDTH, HEIGHT, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            hand_center_y = (y_min + y_max) // 2
            hand_center_x = (x_min + x_max) // 2
            hand_positions[hand_center_x] = hand_center_y

        left_paddle_y = int((1 - smoothing_factor) * left_paddle_y + smoothing_factor * hand_positions.get(min(hand_positions.keys()), left_paddle_y))
        right_paddle_y = int((1 - smoothing_factor) * right_paddle_y + smoothing_factor * hand_positions.get(max(hand_positions.keys()), right_paddle_y))

    # Ball movement
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= HEIGHT:
        ball_speed_y *= -1

    if (ball_x - ball_radius <= left_paddle_x + paddle_width and left_paddle_y <= ball_y <= left_paddle_y + paddle_height) or \
       (ball_x + ball_radius >= right_paddle_x and right_paddle_y <= ball_y <= right_paddle_y + paddle_height):
        ball_speed_x *= -1

    if ball_x < 0:
        right_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_speed_x *= random.choice((1, -1))
        ball_speed_y *= random.choice((1, -1))
    elif ball_x > WIDTH:
        left_score += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_speed_x *= random.choice((1, -1))
        ball_speed_y *= random.choice((1, -1))

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.rect(screen, RED, (left_paddle_x, left_paddle_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, BLUE, (right_paddle_x, right_paddle_y, paddle_width, paddle_height))
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), ball_radius)

    left_text = font.render(f"Left: {left_score}", True, WHITE)
    right_text = font.render(f"Right: {right_score}", True, WHITE)
    screen.blit(left_text, (WIDTH // 4, 20))
    screen.blit(right_text, (3 * WIDTH // 4, 20))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
