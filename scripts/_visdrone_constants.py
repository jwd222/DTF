from __future__ import annotations

VISEDRONE_TO_YOLO = {
    3: 0,   # car               -> compact_car
    4: 2,   # van               -> van
    5: 3,   # truck             -> truck
    6: 5,   # tricycle          -> rickshaw
    7: 5,   # awning-tricycle   -> rickshaw
    8: 4,   # bus               -> bus
    9: 6,   # motor             -> motorcycle
}

YOLO_NAMES = ["compact_car", "suv", "van", "truck", "bus", "rickshaw", "motorcycle"]

VISDRONE_IGNORE = {0, 1, 2}
