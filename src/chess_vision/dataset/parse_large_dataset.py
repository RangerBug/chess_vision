import json
import cv2
import numpy as np
import os
import uuid

def warp_board(frame, corners, size=640):
    matrix = cv2.getPerspectiveTransform(
        np.float32(corners),
        np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    )
    return cv2.warpPerspective(frame, matrix, (size, size))


def crop_and_save(board, config):
    path = "data/"
    board_size = board.shape[0]
    square_size = board_size // 8

    for row in range(8):
        for col in range(8):
            path = "data/"
            x_start = col * square_size
            y_start = row * square_size
            square = board[y_start:y_start+square_size, x_start:x_start+square_size]
            uid = uuid.uuid4().hex[:8]
            name = f"{chr(ord('A') + col)}{8 - row}"
            sq = config["config"]
            if sq.get(name):
                path += "occupied/"
            else:
                path += "empty/"
            file_name = path + name + f"_{uid}.png"
            cv2.imwrite(file_name, square)

def main():
    dataset_loc = "data/synthetic_data/"

    for i in range(1942):
        img = f"{dataset_loc}{i}.jpg"
        j = f"{dataset_loc}{i}.json"
        with open(j, 'r') as f:
            board = json.load(f)

        image = cv2.imread(img)
        height, width, _ = image.shape
        points = []
        for c in board["corners"]:
            x = int(float(c[0]) * width)
            y = height - int(float(c[1]) * height)
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        warped_image = warp_board(image, points, size=640)
        crop_and_save(warped_image, board)

if __name__ == "__main__":
    main()