from __future__ import annotations

VISEDRONE_TO_YOLO = {
    3: 0,   # car               -> car
    4: 1,   # van               -> van
    5: 2,   # truck             -> truck
    6: 3,   # tricycle          -> rickshaw
    7: 3,   # awning-tricycle   -> rickshaw
    8: 4,   # bus               -> bus
    9: 5,   # motor             -> motorcycle
}

YOLO_NAMES = ["car", "van", "truck", "rickshaw", "bus", "motorcycle"]

VISDRONE_IGNORE = {0, 1, 2}
