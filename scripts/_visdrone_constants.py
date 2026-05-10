from __future__ import annotations

VISEDRONE_TO_YOLO = {
    4: 0,   # car               -> car
    5: 1,   # van               -> van
    6: 2,   # truck             -> truck
    7: 3,   # tricycle          -> rickshaw
    8: 3,   # awning-tricycle   -> rickshaw
    9: 4,   # bus               -> bus
    10: 5,  # motor             -> motorcycle
}

YOLO_NAMES = ["car", "van", "truck", "rickshaw", "bus", "motorcycle"]

VISDRONE_IGNORE = {0, 1, 2, 3, 11}
