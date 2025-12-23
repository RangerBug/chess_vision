# Chess Vision Main Script

import cv2
import numpy as np
import torch
from chess_vision.model.model import SimpleSquareClassifier
from torchvision import transforms
import chess
import chess.pgn
from datetime import datetime

# some GLOBAL VARIABLES
board_points = []
game_board = chess.Board()

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleSquareClassifier(num_classes=2)
model.load_state_dict(torch.load("models/SquareClassifier.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


frame_number = 1
def get_frame(cap):
    global frame_number
    if frame_number >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        frame_number = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    frame_number += 1
    if not ret:
        print("Error: Could not read frame.")
        return None
    return frame


def click_event(event, x, y, flags, param):
    global board_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(board_points) < 4:
            board_points.append([x, y])
            print(f"Point {len(board_points)}: ({x}, {y})")


def warp_board(frame, corners, size=640):
    matrix = cv2.getPerspectiveTransform(
        np.float32(corners),
        np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    )
    return cv2.warpPerspective(frame, matrix, (size, size)), matrix


def get_squares(warped_board):
    board_size = warped_board.shape[0]
    square_size = board_size // 8
    squares = []
    for row in range(8):
        for col in range(8):
            x_start = col * square_size
            y_start = row * square_size
            square = warped_board[y_start:y_start+square_size, x_start:x_start+square_size]
            squares.append((row, col, square))
    return squares


def predict_square(squares):
    imgs = []
    for sq in squares:
        img = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        img = transforms.ToPILImage()(img)
        img = transform(img)
        imgs.append(img)
    
    batch = torch.stack(imgs).to(device) # [64, 3, 224, 224]
    
    with torch.no_grad():
        out = model(batch)
        preds = out.argmax(dim=1)

    return preds.cpu().tolist()  # 0 = empty, 1 = occupied


def overlay_results(frame, results, matrix=None):
    if results is None:
        return frame

    board_size = frame.shape[0]
    square_size = board_size // 8
    for row, col, pred in results:
        text = "Occ" if pred == 1 else "Empty"
        x = col * square_size + square_size // 4
        y = row * square_size + square_size // 2
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    return frame


MOTION_THRESHOLD = 5000 # This is random right now
prev_frame = None
motion_counter = 0
stable_counter = 0
STABLE_FRAMES = 15
def detect_motion(new_frame):
    # Return True when processing should continue
    # Return False when processing should halt
    global prev_frame, motion_counter, stable_counter, STABLE_FRAMES, MOTION_THRESHOLD

    # Reduce noise
    gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    if prev_frame is None:
        prev_frame = gray_frame
        return False

    diff = cv2.absdiff(prev_frame, gray_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    motion_pixels = cv2.countNonZero(thresh)

    if motion_pixels > MOTION_THRESHOLD:
        motion_counter += 1
        stable_counter = 0
        #print("Movement detected…")
        prev_frame = gray_frame
        return False
    else:
        stable_counter += 1

    prev_frame = gray_frame

    # Only evaluate the board when fully stable
    if stable_counter < STABLE_FRAMES:
        print(f"Waiting for board to stabilize... ({stable_counter}/{STABLE_FRAMES})")
        return False
    elif stable_counter >= STABLE_FRAMES:
        return True


def log_move(starts, ends):
    global game_board
    f = "abcdefgh" # Files
    r = "87654321" # Ranks
    move_str = ""
    for s_c, s_r in starts:
        move_str += f"{f[s_c]}{r[s_r]}"
        #break # For now ignore more than one entry
    for e_c, e_r in ends:
        move_str += f"{f[e_c]}{r[e_r]}"
        #break # For now ignore more than one entry
    print(move_str)
    
    try:
        move = chess.Move.from_uci(move_str)
        if move in game_board.legal_moves:
            print("NO WAY WE DID IT")
            game_board.push(move)
        else:
            print("Houston we might have a problemoooo")
    except:
        print("Failed to find move")

def save_game():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"saved_games/game_{timestamp}.pgn"
    game = chess.pgn.Game.from_board(game_board)
    game.headers["Event"] = "Game"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    with open(output_file, "w", encoding="utf-8") as f:
        print(game, file=f)
    print(f"Game Saved -> {output_file}")


def main():
    global board_points
    video_loc = "videos/opening.mp4"
    video_type = "opening"
    #video_loc = "videos/endgame.mp4"
    #video_type = "endgame"

    warped_board = None
    results = []
    process_frame = True
    old_board = [[None for _ in range(8)] for _ in range(8)]
    use_video_file = True

    # Start video capture
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) if not use_video_file else cv2.VideoCapture(video_loc)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Main loop
    while True:

        # Get a frame from the camera
        frame = get_frame(cap) 
        if frame is None:
            print("No frame retrieved, exiting.")
            break
        for pt in board_points:
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)    
        cv2.imshow("Chess Vision - Stream", frame)

        # temp for testing
        if use_video_file:
            if video_type == "opening":
                board_points = [[521, 43], [574, 445], [67, 440], [131, 27]]
            elif video_type == "endgame":
                board_points = [[508, 42], [563, 444], [54, 448], [116, 31]]

        if len(board_points) < 4:
            cv2.setMouseCallback("Chess Vision - Stream", click_event)
        else:
            # Rectify the board
            warped_board, warped_matrix = warp_board(frame, board_points, size=640)        
            
            # Detect motion and only process if the frames are still
            # Returns True to process frame; False to halt processing
            no_motion = detect_motion(warped_board)
            if not no_motion: # If there is motion
                process_frame = True # Set this to true so when motion goes away the next line will allow processing
            if no_motion and process_frame:
                print("Processing...")
                # Get squares
                squares = get_squares(warped_board)
                square_imgs = [square for _, _, square in squares]
                preds = predict_square(square_imgs) # Send all square images to process in a single batch

                # Predictions
                results = []
                for (row, col, _), pred in zip(squares, preds):
                    results.append((row, col, pred))
                
                process_frame = False # Make sure no processing happens until after motion is detected

                # Diff Calculation 
                start_changes, end_changes = [], []
                for row, col, pred in results:
                    old_pred = old_board[row][col]
                    if old_pred != pred:
                        if old_pred == 0 and pred == 1: # empty >> occupied
                            end_changes.append((col, row))
                        elif old_pred == 1 and pred == 0: # occupied >> empty
                            start_changes.append((col, row))
                if start_changes or end_changes:
                    log_move(start_changes, end_changes)
                
                # Second pass to write old board because im scared
                for row, col, pred in results:
                    old_board[row][col] = pred

            # Overlay Prediction Results
            #inverse_matrix = np.linalg.inv(warped_matrix)
            warped_results = overlay_results(warped_board.copy(), results)
            cv2.imshow("Chess Vision - Reprojected", warped_results)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Calibrating Board...\nplease select the board corners")
            board_points = []
        elif key == ord('p'):
            print("Pausing... Press any key to continue")
            cv2.waitKey(-1)
        elif key == ord('e'):
            print("Ending Game")
            save_game()
            break # If dont want to break, need to reset the game_board


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
