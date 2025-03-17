import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

import time

import pygetwindow as gw

# 获取所有窗口标题
windows = gw.getAllTitles()
print("所有窗口标题:", windows)

# 根据窗口标题获取窗口对象
window_title = "UU加速器"  # 替换为你的目标窗口标题
target_window = gw.getWindowsWithTitle(window_title)[0]
target_window.activate()  # 激活窗口
target_window.restore()   # 恢复窗口（如果最小化）
time.sleep(1)  # 等待 1 秒，确保窗口激活

def select_region(event, x, y, flags, param):
    global ref_point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Select Region", image)

import numpy as np
import cv2
import win32gui
import win32api
import win32con
import win32ui

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)  # Image with RGBA channels

    # Convert to RGB by discarding the alpha channel
    img = img[:, :, :3]  # Keep only the first 3 channels (RGB)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# window_size = (0,0,1280,720)
# 获取屏幕截图
# image = grab_screen(window_size)

# 显示截图并选择区域
# cv2.namedWindow("Select Region")
# cv2.setMouseCallback("Select Region", select_region)


