import cv2
import numpy as np
import os
import uuid

frozen_frame = None
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points, frozen_frame
    if frozen_frame is not None and event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Point {len(clicked_points)}: ({x}, {y})")
        else:
            print("Already 4 points selected. Press 's' to save.")

def warp_board(frame, corners, size=640):
    matrix = cv2.getPerspectiveTransform(
        np.float32(corners),
        np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    )
    return cv2.warpPerspective(frame, matrix, (size, size))

def crop_and_save_squares(warped_board, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    board_size = warped_board.shape[0]
    square_size = board_size // 8

    for row in range(8):
        for col in range(8):
            x_start = col * square_size
            y_start = row * square_size
            square = warped_board[y_start:y_start+square_size, x_start:x_start+square_size]

            uid = uuid.uuid4().hex[:8]
            file_name = f"{chr(ord('a') + col)}{8 - row}_{uid}.png"
            path = os.path.join(output_dir, file_name)
            cv2.imwrite(path, square)
    print("Saved 64 squares!")

def main():
    global frozen_frame, clicked_points

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    os.makedirs("data", exist_ok=True)
    cv2.namedWindow("Camera View")
    cv2.setMouseCallback("Camera View", click_event)

    print("Press 'c' to freeze frame and select 4 corners")
    print("Press 's' to save squares after selecting corners")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display_frame = frame.copy() if frozen_frame is None else frozen_frame.copy()

        # Draw selected points
        for pt in clicked_points:
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)

        cv2.imshow("Camera View", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Freeze frame
        if frozen_frame is None and key == ord('c'):
            frozen_frame = frame.copy()
            clicked_points = []
            print("Frame frozen! Click 4 corners: top-left, top-right, bottom-right, bottom-left")

        # Save squares
        elif frozen_frame is not None and key == ord('s'):
            if len(clicked_points) != 4:
                print("Select exactly 4 corners before saving!")
            else:
                warped = warp_board(frozen_frame, clicked_points, size=640)
                crop_and_save_squares(warped, output_dir="data")
                # Reset state
                frozen_frame = None
                clicked_points = []
                print("Squares saved. Back to free camera view. Press 'c' to freeze next frame.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
