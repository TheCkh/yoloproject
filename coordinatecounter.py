import cv2

coordinates = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"Point {len(coordinates)}: ({x}, {y})")
        # Draw point on image
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        if len(coordinates) > 1:
            # Draw line between last 2 points
            cv2.line(img, coordinates[-2], coordinates[-1], (0, 255, 0), 2)
        cv2.imshow('Select Region', img)

# Load video and get framer
cap = cv2.VideoCapture("D:/TestFootage/After Video.MOV")
ret, img = cap.read()
cap.release()

# Show image and collect clicks
cv2.imshow('Select Region', img)
cv2.setMouseCallback('Select Region', click_event)

print("Click points to define your region. Press 'q' when done")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\nYour coordinates:")
print(coordinates)  # Ready to use as speed_line
