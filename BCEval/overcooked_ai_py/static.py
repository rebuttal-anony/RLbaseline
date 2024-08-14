import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_current_dir, "data")
HUMAN_DATA_DIR = os.path.join(DATA_DIR, "human_data")
PLANNERS_DIR = os.path.join(DATA_DIR, "planners")
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")
GRAPHICS_DIR = os.path.join(DATA_DIR, "graphics")
FONTS_DIR = os.path.join(DATA_DIR, "fonts")
TESTING_DATA_DIR = os.path.join(DATA_DIR, "testing")
ALL_LAYOUTS = [filename.split('.')[0] for filename in os.listdir(LAYOUTS_DIR) if '.layout' in filename]


OVERCOOK_ACTION_STR2INT  = {
    "NORTH": 0,
    "SOUTH": 1,
    "EAST": 2,
    "WEST": 3,
    "STAY": 4,
    "INTERACT": 5,
}


OVERCOOK_ACTION_INT2STR  = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "STAY",
    5: "INTERACT",
}