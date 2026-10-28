from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time
from dataclasses import dataclass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

user32.SetProcessDPIAware()


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    pid: int
    title: str
    client_x: int
    client_y: int
    client_w: int
    client_h: int


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _process_name(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value.rsplit("\\", 1)[-1]
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return (
        origin.x,
        origin.y,
        rect.right - rect.left,
        rect.bottom - rect.top,
    )


def _visible_windows() -> list[int]:
    handles: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            handles.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return handles


def find_window(process_name: str, title: str = "") -> WindowInfo | None:
    best: WindowInfo | None = None
    for hwnd in _visible_windows():
        pid = _window_pid(hwnd)
        if _process_name(pid).lower() != process_name.lower():
            continue
        window_title = _window_title(hwnd)
        if title and title.lower() not in window_title.lower():
            continue
        x, y, w, h = _client_rect(hwnd)
        if w <= 0 or h <= 0:
            continue
        info = WindowInfo(hwnd, pid, window_title, x, y, w, h)
        if best is None or info.client_w * info.client_h > best.client_w * best.client_h:
            best = info
    return best


def wait_for_window(process_name: str, title: str = "", timeout_s: float = 10.0) -> WindowInfo:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = find_window(process_name, title)
        if info is not None:
            return info
        time.sleep(0.25)
    raise TimeoutError(f"window for {process_name!r} not found within {timeout_s}s")


def refresh(info: WindowInfo) -> WindowInfo:
    x, y, w, h = _client_rect(info.hwnd)
    return WindowInfo(info.hwnd, info.pid, info.title, x, y, w, h)


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def _alt_tap() -> None:
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)


def focus(hwnd: int) -> bool:
    if is_foreground(hwnd):
        return True
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)

    foreground = user32.GetForegroundWindow()
    current_thread = kernel32.GetCurrentThreadId()
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None) if foreground else current_thread
    )

    attached = False
    try:
        if foreground_thread and foreground_thread != current_thread:
            attached = bool(user32.AttachThreadInput(current_thread, foreground_thread, True))
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(current_thread, foreground_thread, False)

    if is_foreground(hwnd):
        return True

    _alt_tap()
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if is_foreground(hwnd):
            return True
        time.sleep(0.05)
    return is_foreground(hwnd)


def is_window_alive(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def client_size(hwnd: int) -> tuple[int, int]:
    x, y, w, h = _client_rect(hwnd)
    return w, h
