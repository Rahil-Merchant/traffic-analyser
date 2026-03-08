import subprocess
videos = [
    {
        "input": r"C:\Users\EMAAN\Documents\YOLO\project\night testing images\videos\20.00.00-22.00.00[R][0@0][0].dav",
        "start": "01:16:30",
        "output": r"C:\Users\EMAAN\Documents\YOLO\project\night testing images\videos\night_clip3.mp4"
    },
]
DURATION = "60"  # seconds
for v in videos:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", v["start"],
        "-i", v["input"],
        "-t", DURATION,
        "-c", "copy",
        v["output"]
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)