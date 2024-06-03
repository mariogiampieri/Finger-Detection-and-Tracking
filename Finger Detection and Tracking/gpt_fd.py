import cv2
import numpy as np

# Define the color range for detecting the finger (this example uses HSV range for skin color)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Initialize the list to store centroid points
centroids = []


def detect_finger(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask where white represents the colors in the range
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply some morphological transformations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the finger
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None


def main():
    # Open video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the finger and get its centroid
        centroid = detect_finger(frame)
        if centroid:
            # Save the centroid point
            centroids.append(centroid)

            # Draw the centroid on the frame
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow("Finger Tracking", frame)

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
