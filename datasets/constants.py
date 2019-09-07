KITCHEN_OBJECT_CLASS_LIST = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMaker",
    "GarbageCan",
    "Box",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Box",
    "Bowl",
]

BEDROOM_OBJECT_CLASS_LIST = ["HousePlant", "Lamp", "Book", "AlarmClock"]


BATHROOM_OBJECT_CLASS_LIST = ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]

AI2Thor_LIVING_ROOM_OBJECT_CLASS_LIST = ['Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl']

AI2Thor_KITCHEN_OBJECT_CLASS_LIST = ['Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl', 'GarbageCan']

AI2Thor_BEDROOM_OBJECT_CLASS_LIST = ['DeskLamp', 'CellPhone', 'Book', 'AlarmClock']

AI2Thor_BATHROOM_OBJECT_CLASS_LIST = ['Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch']


FULL_OBJECT_CLASS_LIST = (
    KITCHEN_OBJECT_CLASS_LIST
    + LIVING_ROOM_OBJECT_CLASS_LIST
    + BEDROOM_OBJECT_CLASS_LIST
    + BATHROOM_OBJECT_CLASS_LIST
)

AI2Thor_FULL_OBJECT_CLASS_LIST = (
    AI2Thor_KITCHEN_OBJECT_CLASS_LIST
    + AI2Thor_LIVING_ROOM_OBJECT_CLASS_LIST
    + AI2Thor_BEDROOM_OBJECT_CLASS_LIST
    + AI2Thor_BATHROOM_OBJECT_CLASS_LIST
)


MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_UP = "LookUp"
LOOK_DOWN = "LookDown"
DONE = "Done"

DONE_ACTION_INT = 5
GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01
