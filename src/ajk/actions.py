from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time

user32 = ctypes.windll.user32

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

_KEYS: dict[str, tuple[int, bool]] = {
    "ESC": (0x01, False),
    "1": (0x02, False),
    "2": (0x03, False),
    "3": (0x04, False),
    "4": (0x05, False),
    "5": (0x06, False),
    "6": (0x07, False),
    "7": (0x08, False),
    "8": (0x09, False),
    "9": (0x0A, False),
    "0": (0x0B, False),
    "MINUS": (0x0C, False),
    "EQUALS": (0x0D, False),
    "BACKSPACE": (0x0E, False),
    "TAB": (0x0F, False),
    "Q": (0x10, False),
    "W": (0x11, False),
    "E": (0x12, False),
    "R": (0x13, False),
    "T": (0x14, False),
    "Y": (0x15, False),
    "U": (0x16, False),
    "I": (0x17, False),
    "O": (0x18, False),
    "P": (0x19, False),
    "LBRACKET": (0x1A, False),
    "RBRACKET": (0x1B, False),
    "ENTER": (0x1C, False),
    "LCTRL": (0x1D, False),
    "A": (0x1E, False),
    "S": (0x1F, False),
    "D": (0x20, False),
    "F": (0x21, False),
    "G": (0x22, False),
    "H": (0x23, False),
    "J": (0x24, False),
    "K": (0x25, False),
    "L": (0x26, False),
    "SEMICOLON": (0x27, False),
    "APOSTROPHE": (0x28, False),
    "GRAVE": (0x29, False),
    "LSHIFT": (0x2A, False),
    "BACKSLASH": (0x2B, False),
    "Z": (0x2C, False),
    "X": (0x2D, False),
    "C": (0x2E, False),
    "V": (0x2F, False),
    "B": (0x30, False),
    "N": (0x31, False),
    "M": (0x32, False),
    "COMMA": (0x33, False),
    "PERIOD": (0x34, False),
    "SLASH": (0x35, False),
    "RSHIFT": (0x36, False),
    "MULTIPLY": (0x37, False),
    "LALT": (0x38, False),
    "SPACE": (0x39, False),
    "CAPSLOCK": (0x3A, False),
    "F1": (0x3B, False),
    "F2": (0x3C, False),
    "F3": (0x3D, False),
    "F4": (0x3E, False),
    "F5": (0x3F, False),
    "F6": (0x40, False),
    "F7": (0x41, False),
    "F8": (0x42, False),
    "F9": (0x43, False),
    "F10": (0x44, False),
    "F11": (0x57, False),
    "F12": (0x58, False),
    "UP": (0x48, True),
    "LEFT": (0x4B, True),
    "RIGHT": (0x4D, True),
    "DOWN": (0x50, True),
    "HOME": (0x47, True),
    "END": (0x4F, True),
    "PAGEUP": (0x49, True),
    "PAGEDOWN": (0x51, True),
    "INSERT": (0x52, True),
    "DELETE": (0x53, True),
    "NUMLOCK": (0x45, True),
    "RCTRL": (0x1D, True),
    "RALT": (0x38, True),
}

_VK: dict[str, int] = {
    "ESC": 0x1B,
    "ENTER": 0x0D,
    "SPACE": 0x20,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "LSHIFT": 0xA0,
    "RSHIFT": 0xA1,
    "LCTRL": 0xA2,
    "RCTRL": 0xA3,
    "LALT": 0xA4,
    "RALT": 0xA5,
    "TAB": 0x09,
    "BACKSPACE": 0x08,
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def vk_for(key: str) -> int:
    name = key.upper()
    if name in _VK:
        return _VK[name]
    if len(name) == 1:
        return user32.VkKeyScanW(ord(name)) & 0xFF
    if name.startswith("F") and name[1:].isdigit():
        return 0x6F + int(name[1:])
    raise KeyError(f"unknown key {key!r}")


class Keyboard:
    def __init__(self, tap_ms: int = 60) -> None:
        self.tap_ms = tap_ms
        self._held: set[str] = set()
        self.events_sent = 0

    def _send(self, key: str, key_up: bool) -> None:
        name = key.upper()
        if name not in _KEYS:
            raise KeyError(f"unknown key {key!r}")
        scan, extended = _KEYS[name]
        flags = KEYEVENTF_SCANCODE
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if key_up:
            flags |= KEYEVENTF_KEYUP
        event = _INPUT(
            type=INPUT_KEYBOARD,
            u=_INPUTUNION(
                ki=_KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=None)
            ),
        )
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))
        self.events_sent += 1

    def down(self, key: str) -> None:
        if key.upper() not in self._held:
            self._send(key, False)
            self._held.add(key.upper())

    def up(self, key: str) -> None:
        if key.upper() in self._held:
            self._send(key, True)
            self._held.discard(key.upper())

    def tap(self, key: str, ms: int | None = None) -> None:
        self.down(key)
        time.sleep((ms if ms is not None else self.tap_ms) / 1000.0)
        self.up(key)

    def release_all(self) -> None:
        for key in list(self._held):
            self.up(key)

    def held(self) -> set[str]:
        return set(self._held)

    def sync(self, keys: set[str]) -> None:
        for key in keys:
            name = key.upper()
            if name not in self._held:
                self._send(key, True)
        for key in list(self._held):
            self._send(key, False)


def key_is_down(key: str) -> bool:
    return bool(user32.GetAsyncKeyState(vk_for(key)) & 0x8000)


def wait_for_key_release(key: str, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not key_is_down(key):
            return True
        time.sleep(0.01)
    return not key_is_down(key)
