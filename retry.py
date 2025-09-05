import sys
import ctypes

# Hide the console window immediately
ctypes.windll.kernel32.FreeConsole()
hwnd = ctypes.windll.kernel32.GetConsoleWindow()
if hwnd:
    ctypes.windll.user32.ShowWindow(hwnd, 0)  # 0 = SW_HIDE

import threading
import time
import numpy as np
import cv2
import pyautogui
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QColorDialog, QTabWidget, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QColor, QPainter, QPen, QFont
from PyQt6.QtCore import Qt, QTimer

# Windows constants
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
GWL_EXSTYLE = -20

def make_window_click_through(hwnd):
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

def send_relative_mouse_move(dx, dy):
    dx = int(np.clip(dx, -30, 30))
    dy = int(np.clip(dy, -30, 30))
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)

def send_left_click():
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)

def send_jump():
    ctypes.windll.user32.keybd_event(0x20, 0, 0, 0)  # Space down
    ctypes.windll.user32.keybd_event(0x20, 0, 2, 0)  # Space up

def get_async_key_state(key_code):
    return ctypes.windll.user32.GetAsyncKeyState(key_code) & 0x8000

import mss

PURPLE_COLOR = QColor(138, 43, 226)  # #8A2BE2

class Overlay(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hwnd = int(self.winId())
        make_window_click_through(hwnd)

        self.screen_width, self.screen_height = pyautogui.size()
        self.center_x, self.center_y = self.screen_width // 2, self.screen_height // 2

        import torch
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', source='github', force_reload=True)

        self.detections = []
        self.detections_lock = threading.Lock()

        threading.Thread(target=self.detection_loop, daemon=True).start()

        self.right_mouse_held = False
        self.left_mouse_held = False

        from pynput import mouse
        self.mouse_listener = mouse.Listener(on_click=self.mouse_click)
        self.mouse_listener.start()

        self.target_lock = TargetLock()
        threading.Thread(target=self.aimbot_loop, daemon=True).start()
        threading.Thread(target=self.triggerbot_loop, daemon=True).start()
        threading.Thread(target=self.anti_recoil_loop, daemon=True).start()
        threading.Thread(target=self.bhop_loop, daemon=True).start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.frame_count = 0
        self.fps = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.calculate_fps)
        self.fps_timer.start(1000)

    def calculate_fps(self):
        self.fps = self.frame_count
        self.frame_count = 0

    def mouse_click(self, x, y, button, pressed):
        from pynput import mouse
        if button == mouse.Button.right:
            self.right_mouse_held = pressed
            if not pressed:
                self.target_lock.clear()
        elif button == mouse.Button.left:
            self.left_mouse_held = pressed

    def detection_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while True:
                sct_img = sct.grab(monitor)
                img_np = np.array(sct_img)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
                results = self.model(img_rgb)
                detections = []
                scale_x = self.screen_width / sct_img.width
                scale_y = self.screen_height / sct_img.height
                for *box, conf, cls in results.xyxy[0]:
                    if float(conf) >= 0.4:
                        x1, y1, x2, y2 = map(int, [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y])
                        detections.append([x1, y1, x2, y2, float(conf)])
                with self.detections_lock:
                    self.detections = detections
                time.sleep(0.03)

    def aimbot_loop(self):
        while True:
            if self.state['aimbot'] and self.right_mouse_held:
                with self.detections_lock:
                    if self.target_lock.target is None and self.target_lock.can_lock():
                        closest = None
                        closest_dist = self.state['fov_radius']
                        for box in self.detections:
                            x1, y1, x2, y2, _ = box
                            cx = (x1 + x2) / 2
                            cy = y1 + (y2 - y1) * 0.20
                            dist = ((cx - self.center_x) ** 2 + (cy - self.center_y) ** 2) ** 0.5
                            if dist <= closest_dist:
                                closest = (cx, cy)
                                closest_dist = dist
                        if closest:
                            self.target_lock.lock(closest)
                            dx = closest[0] - self.center_x
                            dy = closest[1] - self.center_y
                            send_relative_mouse_move(dx, dy)
                            send_left_click()
            time.sleep(0.001)

    def triggerbot_loop(self):
        while True:
            if self.state['triggerbot']:
                with self.detections_lock:
                    for box in self.detections:
                        x1, y1, x2, y2, _ = box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        dist = ((cx - self.center_x) ** 2 + (cy - self.center_y) ** 2) ** 0.5
                        if dist < self.state['trigger_radius']:
                            send_left_click()
                            break
            time.sleep(0.001)

    def anti_recoil_loop(self):
        while True:
            if self.left_mouse_held and self.state['anti_recoil']:
                send_relative_mouse_move(0, 3)
            time.sleep(0.01)

    def bhop_loop(self):
        while True:
            if self.state['bhop'] and get_async_key_state(0x20):
                send_jump()
                time.sleep(0.2)
            time.sleep(0.01)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.state['fov']:
            pen = QPen(PURPLE_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawEllipse(
                int(self.center_x - self.state['fov_radius']),
                int(self.center_y - self.state['fov_radius']),
                int(self.state['fov_radius'] * 2),
                int(self.state['fov_radius'] * 2)
            )

        if self.state['esp']:
            pen = QPen(PURPLE_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            with self.detections_lock:
                for box in self.detections:
                    x1, y1, x2, y2, _ = box
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        if self.state['crosshair']:
            pen = QPen(PURPLE_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            size = 10
            painter.drawLine(self.center_x - size, self.center_y, self.center_x + size, self.center_y)
            painter.drawLine(self.center_x, self.center_y - size, self.center_x, self.center_y + size)

        painter.end()

class TargetLock:
    def __init__(self):
        self.target = None
        self.locked_time = 0
        self.cooldown = 1.0
    def can_lock(self):
        return time.time() - self.locked_time > self.cooldown
    def lock(self, target):
        self.target = target
        self.locked_time = time.time()
    def clear(self):
        self.target = None

class ControlPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(50, 50, 500, 400)
        self.old_pos = None

        main_layout = QVBoxLayout(self)

        # Add "Aorist AI" label at top-left with minimal border
        self.logo_label = QLabel("Aorist AI")
        font = QFont("Arial", 24)
        font.setWeight(QFont.Weight.Bold)
        self.logo_label.setFont(font)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.logo_label.setStyleSheet("""
            color: #4B0082;
            border: none;  /* minimal border */
            padding: 4px;
        """)
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(4)
        glow.setColor(QColor(138, 43, 226, 200))
        glow.setOffset(0)
        self.logo_label.setGraphicsEffect(glow)
        self.logo_label.setFixedWidth(150)
        main_layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Create the tab widget with style that removes default borders
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget {{
                border: none;
            }}
            QTabBar::tab {{
                background: #300052; 
                color: white;
                padding: 10px;
                border-radius: 2px;
            }}
            QTabBar::tab:selected {{
                background: #4B0082;
            }}
        """)
        main_layout.addWidget(self.tabs)

        # --- General Tab ---
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(8)

        self.toggle_buttons = {}
        for key in ['aimbot', 'triggerbot', 'anti_recoil', 'fov', 'crosshair', 'esp', 'bhop']:
            btn = QPushButton(key.replace('_', ' ').title())
            btn.setCheckable(True)
            btn.setChecked(self.state[key])
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #300052;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }}
                QPushButton:checked {{
                    background-color: {PURPLE_COLOR.name()};
                }}
                QPushButton:hover {{
                    background-color: #6A0DAD;
                }}
            """)
            btn.clicked.connect(lambda checked, k=key: self.toggle_feature(k, checked))
            self.toggle_buttons[key] = btn
            toggle_layout.addWidget(btn)
        general_layout.addLayout(toggle_layout)

        def create_slider(label_text, key, min_v, max_v, default):
            lbl = QLabel(f"{label_text}: {default:.2f}")
            lbl.setStyleSheet("color: white; font-weight: bold;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_v * 100))
            slider.setMaximum(int(max_v * 100))
            slider.setValue(int(default * 100))
            slider.valueChanged.connect(lambda val, lbl=lbl, k=key: self.slider_change(val, lbl, k))
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 8px;
                    background: #444;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #8A2BE2;
                    border: none;
                    border-radius: 4px;
                    width: 20px;
                    margin: -6px 0;
                }
                QSlider::sub-page:horizontal {
                    background: #8A2BE2;
                    border-radius: 4px;
                }
            """)
            general_layout.addWidget(lbl)
            general_layout.addWidget(slider)
            return slider

        self.sensitivity_slider = create_slider('Sensitivity', 'sensitivity', 1, 5, self.state['sensitivity'])
        self.smoothing_slider = create_slider('Smoothing', 'smoothing', 0, 1, self.state['smoothing'])
        self.fov_radius_slider = create_slider('FOV Radius', 'fov_radius', 10, 300, self.state['fov_radius'])
        self.ai_confidence_slider = create_slider('AI Confidence', 'ai_confidence', 0.1, 1.0, self.state['ai_confidence'])

        # --- Misc Tab ---
        misc_tab = QWidget()
        misc_layout = QVBoxLayout(misc_tab)

        color_layout = QHBoxLayout()

        def create_color_btn(label, key):
            btn = QPushButton(label)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #4B0082;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: #6A0DAD;
                }}
            """)
            btn.clicked.connect(lambda _, k=key: self.change_color(k))
            color_layout.addWidget(btn)

        create_color_btn('FOV Color', 'fov_color')
        create_color_btn('ESP Color', 'esp_color')
        create_color_btn('Crosshair Color', 'crosshair_color')

        misc_layout.addLayout(color_layout)
        misc_layout.addStretch()

        self.tabs.addTab(general_tab, "General")
        self.tabs.addTab(misc_tab, "Misc")

        self.setStyleSheet("background-color: black;")

    def toggle_feature(self, key, checked):
        self.state[key] = checked

    def slider_change(self, val, lbl, key):
        value = val / 100.0
        lbl.setText(f"{lbl.text().split(':')[0]}: {value:.2f}")
        if key == 'sensitivity':
            self.state['sensitivity'] = value
        elif key == 'smoothing':
            self.state['smoothing'] = value
        elif key == 'fov_radius':
            self.state['fov_radius'] = int(value)
        elif key == 'ai_confidence':
            self.state['ai_confidence'] = value

    def change_color(self, key):
        color = QColorDialog.getColor()
        if color.isValid():
            self.state[key] = color

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

# Better looking, cleaner watermark with glow
class Watermark(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(160, 80)
        self.move(10, 10)
        self.fps = 0
        self.dev_text = "Dev : BG"

        # Add glow effect for a cleaner look
        glow_effect = QGraphicsDropShadowEffect()
        glow_effect.setBlurRadius(10)
        glow_effect.setColor(QColor(138, 43, 226, 180))
        glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(glow_effect)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update)
        self.fps_timer.start(1000)

        self.frame_count = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background rectangle
        painter.fillRect(0, 0, self.width(), self.height(), QColor(0, 0, 0, 150))

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 10)
        painter.setFont(font)

        # Draw texts with some spacing
        painter.drawText(10, 20, "Aorist AI")
        fps_text = f"FPS: {self.fps}"
        painter.drawText(10, 40, fps_text)
        painter.drawText(10, 60, self.dev_text)
        painter.end()

    def update_fps(self, fps_value):
        self.fps = int(fps_value)
        self.update()

# Main execution
if __name__ == "__main__":
    import torch
    from pynput import mouse

    app = QApplication(sys.argv)

    # Initial state
    state = {
        'aimbot': False,
        'triggerbot': False,
        'anti_recoil': False,
        'fov': False,
        'crosshair': False,
        'esp': False,
        'bhop': False,
        'fov_radius': 80,
        'sensitivity': 2.11,
        'smoothing': 0.170,
        'ai_confidence': 0.7,
        'fov_color': QColor(255, 255, 0, 100),
        'esp_color': QColor(255, 0, 0, 200),
        'crosshair_color': QColor(0, 255, 255, 200),
        'trigger_radius': 50,
    }

    control_panel = ControlPanel(state)
    control_panel.show()

    overlay = Overlay(state)
    overlay.showFullScreen()

    watermark = Watermark()
    watermark.show()

    def update_fps():
        fps_value = overlay.fps
        watermark.update_fps(fps_value)

    fps_update_timer = QTimer()
    fps_update_timer.timeout.connect(update_fps)
    fps_update_timer.start(1000)

    def check_insert_press():
        if ctypes.windll.user32.GetAsyncKeyState(0x2D) & 0x8000:
            if control_panel.isVisible():
                control_panel.hide()
            else:
                control_panel.show()
            time.sleep(0.3)

    toggle_timer = QTimer()
    toggle_timer.timeout.connect(check_insert_press)
    toggle_timer.start(200)

    sys.exit(app.exec())
