# import subprocess

# # 1st clip: 0 to 5 minutes (300 seconds)
# subprocess.run([
#     "ffmpeg", "-i", "C:/Users/EMAAN/Documents/YOLO/DAV Videos/11.00.00-13.00.00[R][0@0][1]_416x416_noaudio.mp4", "-ss", "00:00:00", "-t", "00:05:00",
#     "-c", "copy", "part1.mp4"
# ])

# # 2nd clip: 5 to 10 minutes
# subprocess.run([
#     "ffmpeg", "-i", "C:/Users/EMAAN/Documents/YOLO/DAV Videos/11.00.00-13.00.00[R][0@0][1]_416x416_noaudio.mp4", "-ss", "00:05:00", "-t", "00:05:00",
#     "-c", "copy", "part2.mp4"
# ])


import cv2

cap = cv2.VideoCapture("C:/Users/EMAAN/Documents/YOLO/DAV Videos/part1.mp4")
ret, frame = cap.read()
print(ret)