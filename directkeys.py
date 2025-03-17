import ctypes
import time

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F
SPACE = 0x39  # 空格键

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21
Z = 0x2C
X = 0x2D

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# 鼠标按键的键码
MOUSEEVENTF_RIGHTDOWN = 0x0008  # 右键按下
MOUSEEVENTF_RIGHTUP = 0x0010    # 右键释放


# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions


def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def PressRightMouse():
    """
    模拟按下鼠标右键。
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)  # 0 表示鼠标输入
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseRightMouse():
    """
    模拟释放鼠标右键。
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)  # 0 表示鼠标输入
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def move_left(duration=1):
    """
    模拟按下 A 键（向左移动）。
    :param duration: 按键按下的持续时间（秒）。
    """
    PressKey(A)  # 按下 A 键
    time.sleep(duration)  # 保持按下状态
    ReleaseKey(A)  # 释放 A 键

def move_up(duration=1):
    """
    模拟按下 W 键（向上移动）。
    :param duration: 按键按下的持续时间（秒）。
    """
    PressKey(W)  # 按下 W 键
    time.sleep(duration)  # 保持按下状态
    ReleaseKey(W)  # 释放 W 键

def move_down(duration=1):
    """
    模拟按下 S 键（向下移动）。
    :param duration: 按键按下的持续时间（秒）。
    """
    PressKey(S)  # 按下 S 键
    time.sleep(duration)  # 保持按下状态
    ReleaseKey(S)  # 释放 S 键

def move_right(duration=1):
    """
    模拟按下 D 键（向右移动）。
    :param duration: 按键按下的持续时间（秒）。
    """
    PressKey(D)  # 按下 D 键
    time.sleep(duration)  # 保持按下状态
    ReleaseKey(D)  # 释放 D 键


def dodge():#闪避
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(SPACE)
    #time.sleep(0.1)

def mainskill():
    """
    主要技能函数，绑定鼠标右键。
    """
    print("使用主要技能...")
    PressRightMouse()  # 按下鼠标右键
    time.sleep(0.1)    # 保持按下状态（根据需要调整时间）
    ReleaseRightMouse()  # 释放鼠标右键


def secondskill():
    PressKey(Q)
    time.sleep(0.1)
    ReleaseKey(Q)


def thirdskill():
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)


def pickup():
    PressKey(Z)
    time.sleep(0.1)
    ReleaseKey(Z)


def interact():
    PressKey(X)
    time.sleep(0.1)
    ReleaseKey(X)