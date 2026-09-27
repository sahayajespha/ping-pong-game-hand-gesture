import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define dataset path
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# CSV File to store landmark data
csv_file = os.path.join(dataset_path, "hand_landmarks_dataset.csv")

# Initialize CSV file with headers if not exists
if not os.path.exists(csv_file):
    columns = ["Gesture", "Hand"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# Function to extract hand landmarks and save them
def save_hand_landmarks(hand_landmarks, hand_label, gesture):
    if not gesture:
        return  # No gesture selected, don't save
    
    # Extract landmarks
    landmarks = [gesture, hand_label]  # Store Gesture & Hand (LEFT/RIGHT)
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    # Save to CSV
    df = pd.DataFrame([landmarks])
    df.to_csv(csv_file, mode='a', header=False, index=False)
    print(f"Saved {hand_label} hand for {gesture}")

# Open webcam
cap = cv2.VideoCapture(0)
current_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip image for better alignment
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_idx].classification[0].label.upper()  # LEFT or RIGHT
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if current_gesture:
                save_hand_landmarks(hand_landmarks, hand_label, current_gesture)
    
    # Display selected gesture
    if current_gesture:
        cv2.putText(frame, f"Recording: {current_gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press U (Up), D (Down), S (Stay), Q (Quit)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Hand Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        current_gesture = "Move_Up"
    elif key == ord('d'):
        current_gesture = "Move_Down"
    elif key == ord('s'):
        current_gesture = "Stay"
    elif key == ord('q'):
        break
    else:
        current_gesture = None  # Stop recording if no key is pressed

cap.release()
cv2.destroyAllWindows()
