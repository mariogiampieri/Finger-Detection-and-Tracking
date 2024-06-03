import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the list to store centroid points
centroids = []


def main():
    # Open video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Calculate the centroid of the hand landmarks
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                centroid = (
                    int(sum(x_coords) / len(x_coords) * frame.shape[1]),
                    int(sum(y_coords) / len(y_coords) * frame.shape[0]),
                )

                # Save the centroid point
                centroids.append(centroid)

                # Draw the centroid on the frame
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the centroids to a file or render them
    with open("centroids.txt", "w") as f:
        for point in centroids:
            f.write(f"{point[0]},{point[1]}\n")

    print("Centroids saved to centroids.txt")


if __name__ == "__main__":
    main()
