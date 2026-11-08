"""Simple Tkinter viewer for rendered multi-camera tracking results."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path

import cv2

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


class App:
    """Display up to four camera streams with report bounding boxes."""

    def __init__(self, root: tk.Tk, video_dir: str, report: str) -> None:
        if ImageTk is None:
            raise RuntimeError("Pillow is required to display frames")
        self.root = root
        self.root.title("Cross-camera tracking")
        self.report = json.loads(Path(report).read_text(encoding="utf-8"))
        self.captures = [
            cv2.VideoCapture(str(path))
            for path in sorted(Path(video_dir).glob("*.mp4"))
        ]
        self.frame_number = 0
        self.paused = False
        self.labels = []

        toolbar = tk.Frame(root)
        toolbar.pack(fill="x")
        tk.Label(
            toolbar,
            text=(
                f"Tracks: {self.report['total_tracks']} | " f"{self.report['detector']}"
            ),
        ).pack(side="left", padx=10, pady=8)
        tk.Button(toolbar, text="Play / pause", command=self.toggle).pack(
            side="right", padx=10
        )

        body = tk.Frame(root)
        body.pack(fill="both", expand=True)
        for index in range(min(4, len(self.captures))):
            body.grid_rowconfigure(index // 2, weight=1)
            body.grid_columnconfigure(index % 2, weight=1)
            label = tk.Label(body)
            label.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=4,
                pady=4,
            )
            self.labels.append(label)
        root.after(40, self.tick)

    def toggle(self) -> None:
        self.paused = not self.paused

    def tick(self) -> None:
        if not self.paused:
            for index, capture in enumerate(self.captures[: len(self.labels)]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, (640, 480))
                for track in self.report["tracks"]:
                    for item in track.get("boxes", []):
                        if item["camera"] != index + 1:
                            continue
                        if abs(item["frame"] - self.frame_number) > 12:
                            continue
                        x1, y1, x2, y2 = item["box"]
                        scale_x = 640 / 2592
                        scale_y = 480 / 1944
                        points = [
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                        ]
                        cv2.rectangle(
                            frame,
                            (points[0], points[1]),
                            (points[2], points[3]),
                            (0, 220, 80),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"ID {track['global_id']}",
                            (points[0], max(20, points[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 220, 80),
                            2,
                        )
                photo = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                self.labels[index].configure(image=photo)
                self.labels[index].image = photo
            self.frame_number += 1
        self.root.after(40, self.tick)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--report", default="outputs/report.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    App(tk.Tk(), arguments.videos, arguments.report).root.mainloop()
