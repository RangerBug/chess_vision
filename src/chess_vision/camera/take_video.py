import cv2

def record_video(
    output_path="endgame.mp4",
    fps=30,
    frame_width=640,
    frame_height=480,
    camera_index=0,
    max_seconds=10
):
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Check camera open
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording... Press 'q' to quit early.")

    frame_count = 0
    max_frames = fps * max_seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        out.write(frame)

        # Show preview (optional)
        cv2.imshow("Recording", frame)

        # Quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Stop when time is up
        frame_count += 1
        if frame_count >= max_frames:
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    record_video(max_seconds=30)  # Record 30 seconds
