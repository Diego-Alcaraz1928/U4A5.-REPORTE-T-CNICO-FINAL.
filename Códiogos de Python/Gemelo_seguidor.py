#!/usr/bin/env python3
"""
==============================================================================
 GEMELO DIGITAL — SEGUIDOR DE LÍNEA PID (PyQt5)
==============================================================================
Aplicación de escritorio que simula un seguidor de línea con:
  • 4 sensores QTR analógicos equidistantes con penumbra
  • 2 motores TT 48:1 con curva torque-velocidad y back-EMF
  • Control PID por posición ponderada
  • Pistas procedurales o cargadas desde PNG (10 px = 1 cm, líneas 1.9 cm)
  • Visualización 2D en vivo con trayectoria
  • Telemetría en tiempo real (error, PWM, componentes PID)
  • Importar/exportar presets JSON y telemetría CSV
  • Tooltips, sliders, validaciones y métricas de desempeño

Ejecuta:  python3 line_follower_pyqt.py
Requiere: PyQt5, numpy, matplotlib, Pillow
==============================================================================
"""
# ----------------------------------------------------------------------------
# IMPORTANTE: si PySide6 también está instalado, matplotlib puede elegirlo
# automáticamente y mezclarlo con PyQt5, lo que rompe addTab() y otros métodos
# por incompatibilidad de tipos. Forzamos PyQt5 ANTES de importar matplotlib.
# ----------------------------------------------------------------------------
import os
os.environ["QT_API"] = "PyQt5"
os.environ["MPLBACKEND"] = "Qt5Agg"

import sys
import json
import csv
import copy
import time
from collections import deque

import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF, QSize
from PyQt5.QtGui import (QPixmap, QPainter, QColor, QPen, QBrush, QFont,
                         QPolygonF, QImage, QTransform, QIcon)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QSlider, QComboBox,
    QFileDialog, QGroupBox, QTabWidget, QSplitter, QFrame, QSizePolicy,
    QScrollArea, QToolTip, QStatusBar, QMessageBox, QStyleFactory,
)

from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Qt5Agg", force=True)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================================
# CONSTANTS
# ============================================================================
PX_PER_M = 1000  # 1000 px = 1 m  (10 px = 1 cm)
LINE_WIDTH_M = 0.019
G = 9.81

DEFAULT_CONFIG = {
    'robot': {
        'mass': 0.300, 'wheelDiameter': 0.067, 'wheelBase': 0.10,
        'sensorSpacing': 0.012, 'sensorAxisDistance': 0.07,
        'rollingFriction': 0.04, 'lateralFriction': 8.0,
        'chassisLength': 0.12, 'chassisWidth': 0.10,
    },
    'motor': {
        'voltage': 11.1, 'noLoadRPM_at_nominal': 370,
        'stallTorque': 0.145, 'nominalVoltage': 11.1,
    },
    'sensor': {
        'samplingRate': 200, 'sensorRadius': 0.0015, 'samplingPoints': 9,
    },
    'pid': {
        'kp': 0.0008, 'ki': 0.0, 'kd': 0.0015,
        'baseSpeed': 0.55, 'maxSpeed': 1.0,
        'integralLimit': 500, 'derivativeFilter': 0.7,
        'deadZone': 0.0, 'deadZoneEnabled': False,
    },
    'sim': {
        'physicsDt': 0.001, 'telemetryDecimation': 5, 'simSpeed': 1.0,
    },
    'start': {'x': 0.40, 'y': 0.501, 'theta': 0.0},
}


# ============================================================================
# TRACK
# ============================================================================
class Track:
    GENERATED_KINDS = ['oval', 'square', 'figure8', 'scurve', 'race', 'complex']

    # Calibradas empíricamente: la fila de sensores cae centrada sobre la línea
    START_POSE = {
        'oval':    (0.75, 0.840, 0.0),
        'square':  (0.50, 0.109, 0.0),
        'figure8': (0.412, 0.834, 0.0),
        'scurve':  (0.10, 0.591, 0.0),
        'race':    (0.20, 0.818, 0.0),
        'complex': (0.40, 0.501, 0.0),
    }

    def __init__(self, source='complex', w=1500, h=1000):
        if isinstance(source, str) and source in self.GENERATED_KINDS:
            self.image = self._generate(source, w, h)
            self.kind = source
        elif isinstance(source, str):
            self.image = Image.open(source).convert('L')
            self.kind = 'file'
        else:
            self.image = source.convert('L') if isinstance(source, Image.Image) else None
            self.kind = 'image'
        self.w, self.h = self.image.size
        self.data = np.array(self.image, dtype=np.float32) / 255.0

    @staticmethod
    def _generate(kind, w, h):
        img = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(img)
        lw = int(LINE_WIDTH_M * PX_PER_M)
        if kind == 'oval':
            draw.ellipse([w*0.10, h*0.15, w*0.90, h*0.85], outline=0, width=lw)
        elif kind == 'square':
            draw.rounded_rectangle([100, 100, w-100, h-100],
                                   radius=80, outline=0, width=lw)
        elif kind == 'figure8':
            draw.ellipse([w*0.05, h*0.15, w*0.50, h*0.85], outline=0, width=lw)
            draw.ellipse([w*0.50, h*0.15, w*0.95, h*0.85], outline=0, width=lw)
        elif kind == 'scurve':
            pts = [(100 + (w-200)*i/199,
                    h/2 + 200*np.sin(2*np.pi*(i/199)*1.5)) for i in range(200)]
            for i in range(len(pts)-1):
                draw.line([pts[i], pts[i+1]], fill=0, width=lw)
        elif kind == 'race':
            way = [(200,800),(400,850),(600,700),(800,800),
                   (1000,600),(1200,700),(1300,500),(1100,300),
                   (900,400),(700,200),(500,250),(300,400),(200,800)]
            for i in range(len(way)-1):
                draw.line([way[i], way[i+1]], fill=0, width=lw)
        elif kind == 'complex':
            draw.rounded_rectangle([100, 200, w-100, h-200],
                                   radius=200, outline=0, width=lw)
            draw.line([(300, h//2), (w-300, h//2)], fill=0, width=lw)
        return img

    def darkness(self, x_m, y_m):
        px, py = int(x_m * PX_PER_M), int(y_m * PX_PER_M)
        if 0 <= px < self.w and 0 <= py < self.h:
            return 1.0 - float(self.data[py, px])
        return 0.0


# ============================================================================
# SIMULATOR (física validada)
# ============================================================================
class Simulator:
    SENSOR_OFFSETS_K = np.array([-1.5, -0.5, 0.5, 1.5])
    PID_WEIGHTS = np.array([-3000., -1000., 1000., 3000.])

    def __init__(self, config, track):
        self.cfg = copy.deepcopy(config)
        self.track = track
        self.reset()

    def reset(self):
        s = self.cfg['start']
        self.x, self.y, self.theta = s['x'], s['y'], s['theta']
        self.vx = self.vy = self.omega = 0.0
        self.pwm_l = self.pwm_r = 0.0
        self.error = self.error_prev = 0.0
        self.integral = self.d_filt = 0.0
        self.first_pid = True
        self.P = self.I = self.D = 0.0
        self.sensors = np.zeros(4)
        self.t = 0.0
        self.distance = 0.0
        self.on_line = False
        self.estado = 'centrado'
        self.corr_blocked = False
        self.trail = deque(maxlen=4000)
        self.history = []

    def update_config(self, cfg):
        """Aplica cambios de config sin resetear el estado del robot."""
        self.cfg = copy.deepcopy(cfg)

    def sense(self):
        c = self.cfg
        offsets = self.SENSOR_OFFSETS_K * c['robot']['sensorSpacing']
        d = c['robot']['sensorAxisDistance']
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        r = c['sensor']['sensorRadius']
        N = c['sensor']['samplingPoints']
        out = np.zeros(4)
        for i, ly in enumerate(offsets):
            wx = self.x + d * cos_t - ly * sin_t
            wy = self.y + d * sin_t + ly * cos_t
            acc = self.track.darkness(wx, wy)
            for k in range(1, N):
                ang = 2 * np.pi * k / max(1, N - 1)
                rr = r * 0.66
                acc += self.track.darkness(wx + np.cos(ang)*rr,
                                           wy + np.sin(ang)*rr)
            out[i] = max(0, min(1023, round(acc / N * 1023)))
        return out

    def pid_step(self, sensors, dt_pid):
        ssum = sensors.sum()
        if ssum < 50:
            self.estado = 'linea_perdida'
            return None
        error = float((sensors * self.PID_WEIGHTS).sum() / ssum)
        c = self.cfg['pid']
        if self.first_pid:
            self.error_prev = error
            self.first_pid = False
        else:
            self.error_prev = self.error
        self.error = error

        # ── Clasificar estado ────────────────────────────────────────────
        abs_err = abs(error)
        abs_prev = abs(self.error_prev)
        if abs_err < 300:
            self.estado = 'centrado'
        elif abs_err >= abs_prev:
            self.estado = 'desviado'
        else:
            self.estado = 'recuperacion'

        # ── Zona muerta ──────────────────────────────────────────────────
        dz = c.get('deadZone', 0.0)
        dz_on = c.get('deadZoneEnabled', False)
        self.corr_blocked = dz_on and (abs_err < dz)

        # Congelar integrador dentro de la zona muerta para evitar windup
        if not self.corr_blocked:
            self.integral = float(np.clip(self.integral + error * dt_pid,
                                          -c['integralLimit'], c['integralLimit']))
        d_raw = (error - self.error_prev) / dt_pid
        f = c['derivativeFilter']
        self.d_filt = f * self.d_filt + (1 - f) * d_raw
        self.P = c['kp'] * error
        self.I = c['ki'] * self.integral
        self.D = c['kd'] * self.d_filt
        correction = self.P + self.I + self.D

        if self.corr_blocked:
            return 0.0
        return correction

    def physics_step(self, dt):
        c = self.cfg
        wheel_r = c['robot']['wheelDiameter'] / 2
        omega_no_load = c['motor']['noLoadRPM_at_nominal'] * 2 * np.pi / 60
        K_drive = c['motor']['stallTorque']
        K_emf = K_drive / max(omega_no_load, 1e-6)
        m = c['robot']['mass']
        Iz = m * (c['robot']['chassisLength']**2 + c['robot']['chassisWidth']**2) / 12
        half_b = c['robot']['wheelBase'] / 2
        Fn = m * G
        cR = c['robot']['rollingFriction']
        b_lat = c['robot']['lateralFriction']
        v_scale = c['motor']['voltage'] / max(c['motor']['nominalVoltage'], 1e-6)

        vL = self.vx - self.omega * half_b
        vR = self.vx + self.omega * half_b
        wL, wR = vL / wheel_r, vR / wheel_r
        tau_L = K_drive * self.pwm_l * v_scale - K_emf * wL
        tau_R = K_drive * self.pwm_r * v_scale - K_emf * wR
        FL, FR = tau_L / wheel_r, tau_R / wheel_r

        F_drive = FL + FR
        F_roll = cR * Fn * (np.sign(self.vx) if abs(self.vx) > 1e-4 else 0)
        F_y = -b_lat * self.vy
        tau_yaw = (FR - FL) * half_b
        tau_yaw_drag = -b_lat * self.omega * half_b * half_b * 0.5

        ax = (F_drive - F_roll) / m
        ay = F_y / m
        a_om = (tau_yaw + tau_yaw_drag) / Iz

        self.vx += ax * dt
        self.vy += ay * dt
        self.omega += a_om * dt

        if abs(self.pwm_l) < 0.01 and abs(self.pwm_r) < 0.01:
            if abs(self.vx) < 0.005: self.vx = 0
            if abs(self.omega) < 0.05: self.omega = 0

        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        dx = (self.vx * cos_t - self.vy * sin_t) * dt
        dy = (self.vx * sin_t + self.vy * cos_t) * dt
        self.x += dx
        self.y += dy
        self.theta += self.omega * dt
        self.distance += np.hypot(dx, dy)
        self.t += dt

    def step(self, real_dt):
        """Avanza la simulación durante real_dt segundos (ajustado por simSpeed)."""
        c = self.cfg
        sim_dt = real_dt * c['sim']['simSpeed']
        dt = c['sim']['physicsDt']
        n = max(1, int(sim_dt / dt))
        # Limitar para evitar congelamiento si simSpeed * realDt es enorme
        n = min(n, 5000)
        sample_period = 1.0 / c['sensor']['samplingRate']

        if not hasattr(self, '_sample_acc'):
            self._sample_acc = 0.0
        if not hasattr(self, '_trail_acc'):
            self._trail_acc = 0
        rec_every = c['sim']['telemetryDecimation']

        for i in range(n):
            self._sample_acc += dt
            if self._sample_acc >= sample_period:
                self._sample_acc -= sample_period
                self.sensors = self.sense()
                u = self.pid_step(self.sensors, sample_period)
                if u is not None:
                    self.on_line = True
                    base = c['pid']['baseSpeed']
                    mx = c['pid']['maxSpeed']
                    # Clamp [0, max] (no reversa, como un seguidor real)
                    self.pwm_l = float(np.clip(base - u, 0.0, mx))
                    self.pwm_r = float(np.clip(base + u, 0.0, mx))
                else:
                    self.on_line = False
                    self.integral *= 0.95
            self.physics_step(dt)

            self._trail_acc += 1
            if self._trail_acc >= 20:
                self._trail_acc = 0
                self.trail.append((self.x, self.y))

            if i % rec_every == 0:
                self.history.append({
                    't': self.t, 'x': self.x, 'y': self.y, 'theta': self.theta,
                    's0': float(self.sensors[0]), 's1': float(self.sensors[1]),
                    's2': float(self.sensors[2]), 's3': float(self.sensors[3]),
                    'error': self.error, 'P': self.P, 'I': self.I, 'D': self.D,
                    'pwm_l': self.pwm_l, 'pwm_r': self.pwm_r,
                    'vx': self.vx, 'omega': self.omega, 'on_line': self.on_line,
                    'estado': self.estado, 'corr_bloqueada': self.corr_blocked,
                })
        # Cap historial para que no crezca indefinidamente
        if len(self.history) > 20000:
            self.history = self.history[-15000:]

    def metrics(self):
        if not self.history:
            return {}
        errs = np.array([h['error'] for h in self.history]) / 3000.0
        signs = np.sign(errs)
        oscill = int(np.sum((signs[1:] != signs[:-1]) & (signs[:-1] != 0)))
        return {
            'tiempo_s': self.t,
            'distancia_m': self.distance,
            'vel_media_mps': self.distance / self.t if self.t > 0 else 0.0,
            'rms_error': float(np.sqrt(np.mean(errs**2))),
            'max_error': float(np.max(np.abs(errs))),
            'oscilaciones': oscill,
            'porc_en_linea': 100.0 * np.mean([h['on_line'] for h in self.history]),
        }


# ============================================================================
# QT WIDGETS
# ============================================================================
class TrackView(QWidget):
    """Widget que dibuja la pista, la trayectoria y el robot.
    Soporta click para colocar el robot."""
    robotPlaced = pyqtSignal(float, float)  # x_m, y_m

    def __init__(self):
        super().__init__()
        self.track = None
        self.track_pixmap = None
        self.simulator = None
        self.placing = False
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(640, 480))
        self.setStyleSheet("background:#0a0a0b;")
        self._scale = 1.0
        self._ox = 0
        self._oy = 0

    def set_track(self, track):
        self.track = track
        # Convertir PIL -> QPixmap (vía QImage)
        rgb = track.image.convert('RGB')
        qimg = QImage(rgb.tobytes(), rgb.width, rgb.height,
                      rgb.width * 3, QImage.Format_RGB888)
        self.track_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_simulator(self, sim):
        self.simulator = sim
        self.update()

    def set_placing(self, placing):
        self.placing = placing
        self.setCursor(Qt.CrossCursor if placing else Qt.ArrowCursor)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0a0a0b"))
        if self.track_pixmap is None:
            return

        W, H = self.width(), self.height()
        tw, th = self.track_pixmap.width(), self.track_pixmap.height()
        scale = min(W / tw, H / th)
        ox = (W - tw * scale) / 2
        oy = (H - th * scale) / 2
        self._scale, self._ox, self._oy = scale, ox, oy

        target = QRectF(ox, oy, tw * scale, th * scale)
        p.drawPixmap(target, self.track_pixmap, QRectF(0, 0, tw, th))

        # Borde del área
        p.setPen(QPen(QColor(63, 63, 70), 1))
        p.drawRect(target)

        if self.simulator is None:
            return

        # Trayectoria
        s = self.simulator
        if len(s.trail) > 1:
            pen = QPen(QColor(34, 211, 238, 200), 2)
            p.setPen(pen)
            poly = QPolygonF()
            for x_m, y_m in s.trail:
                px = ox + x_m * PX_PER_M * scale
                py = oy + y_m * PX_PER_M * scale
                poly.append(QPointF(px, py))
            p.drawPolyline(poly)

        # Robot
        cfg = s.cfg
        robot_x = ox + s.x * PX_PER_M * scale
        robot_y = oy + s.y * PX_PER_M * scale
        L = cfg['robot']['chassisLength'] * PX_PER_M * scale
        Wp = cfg['robot']['chassisWidth'] * PX_PER_M * scale

        p.save()
        p.translate(robot_x, robot_y)
        p.rotate(np.degrees(s.theta))
        # cuerpo
        p.setBrush(QBrush(QColor(132, 204, 22, 60)))
        p.setPen(QPen(QColor(163, 230, 53), 2))
        p.drawRect(QRectF(-L/4, -Wp/2, L, Wp))
        # flecha de orientación
        p.setPen(QPen(QColor(163, 230, 53), 3))
        p.drawLine(QPointF(0, 0), QPointF(L*0.6, 0))
        # ruedas
        wb_px = cfg['robot']['wheelBase'] * PX_PER_M * scale
        wheel_l = 0.04 * PX_PER_M * scale
        wheel_w = 0.012 * PX_PER_M * scale
        p.setBrush(QColor(39, 39, 42))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(-wheel_l/2, -wb_px/2 - wheel_w, wheel_l, wheel_w))
        p.drawRect(QRectF(-wheel_l/2,  wb_px/2,            wheel_l, wheel_w))
        # sensores
        sd = cfg['robot']['sensorAxisDistance'] * PX_PER_M * scale
        sp = cfg['robot']['sensorSpacing'] * PX_PER_M * scale
        for i, k in enumerate([-1.5, -0.5, 0.5, 1.5]):
            sy = k * sp
            v = s.sensors[i] / 1023.0
            p.setBrush(QColor(int(255 - v*150), int(230 - v*200), int(50 + v*50)))
            p.setPen(QPen(Qt.white, 1))
            r = cfg['sensor']['sensorRadius'] * PX_PER_M * scale * 1.6
            p.drawEllipse(QPointF(sd, sy), r, r)
        # línea entre sensores
        p.setPen(QPen(QColor(163, 230, 53, 120), 1))
        p.drawLine(QPointF(sd, -1.5*sp), QPointF(sd, 1.5*sp))
        p.restore()

        # HUD
        p.setBrush(QColor(10, 10, 12, 220))
        p.setPen(QPen(QColor(34, 211, 238, 100), 1))
        hud = QRectF(12, 12, 230, 138)
        p.drawRect(hud)
        p.setFont(QFont("monospace", 9))
        p.setPen(QColor(161, 161, 170))
        p.drawText(20, 30, f"X: {s.x*100:6.2f} cm   Y: {s.y*100:6.2f} cm")
        p.drawText(20, 46, f"θ: {np.degrees(s.theta):+7.1f}°")
        p.drawText(20, 62, f"v: {s.vx:+.3f} m/s   ω: {s.omega:+.2f} rad/s")
        p.drawText(20, 78, f"PWM L: {s.pwm_l:.2f}   R: {s.pwm_r:.2f}")
        p.drawText(20, 94, f"err: {s.error:+8.0f}")
        p.setPen(QColor(163, 230, 53) if s.on_line else QColor(248, 113, 113))
        p.drawText(20, 112, "● EN LÍNEA" if s.on_line else "✕ LÍNEA PERDIDA")

        # Estado del robot
        estado_colors = {
            'centrado':     QColor(163, 230, 53),
            'desviado':     QColor(248, 113, 113),
            'recuperacion': QColor(251, 191, 36),
            'linea_perdida':QColor(239, 68, 68),
        }
        estado_txt = getattr(s, 'estado', 'centrado')
        p.setPen(estado_colors.get(estado_txt, QColor(161, 161, 170)))
        dz_ind = " 🔒" if getattr(s, 'corr_blocked', False) else ""
        p.drawText(20, 130, f"▸ {estado_txt.upper()}{dz_ind}")

        # Indicador "colocar"
        if self.placing:
            p.setPen(QPen(QColor(34, 211, 238), 2))
            p.drawText(QRectF(0, H-30, W, 30), Qt.AlignCenter,
                       "Click sobre la pista para colocar el robot")

    def mousePressEvent(self, ev):
        if not self.placing or self.track_pixmap is None:
            return
        # convertir pixel pantalla → metros mundo
        x_m = (ev.x() - self._ox) / (self._scale * PX_PER_M)
        y_m = (ev.y() - self._oy) / (self._scale * PX_PER_M)
        if 0 <= x_m * PX_PER_M <= self.track.w and 0 <= y_m * PX_PER_M <= self.track.h:
            self.robotPlaced.emit(x_m, y_m)


class LiveChart(FigureCanvas):
    """Canvas matplotlib con varias series en tiempo real."""
    def __init__(self, title, series, ylim=None, parent=None):
        self.fig = Figure(figsize=(4, 2), tight_layout=True, facecolor='#18181b')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0a0a0b')
        self.ax.tick_params(colors='#71717a', labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color('#3f3f46')
        self.ax.grid(True, alpha=0.2, color='#3f3f46')
        self.ax.set_title(title, color='#a1a1aa', fontsize=9)
        if ylim:
            self.ax.set_ylim(*ylim)
        self.ax.axhline(0, color='#52525b', lw=0.5)

        self.lines = {}
        for name, color in series:
            line, = self.ax.plot([], [], color=color, lw=1.2, label=name)
            self.lines[name] = line
        if len(series) > 1:
            self.ax.legend(loc='upper right', fontsize=7,
                           facecolor='#0a0a0b', edgecolor='#3f3f46',
                           labelcolor='#a1a1aa')
        self.setMinimumHeight(150)

    def update_series(self, t, data):
        if not t:
            return
        for name, ydata in data.items():
            if name in self.lines:
                self.lines[name].set_data(t, ydata)
        tmin, tmax = t[0], t[-1]
        if tmax > tmin:
            self.ax.set_xlim(tmin, tmax)
        # autoscale Y si no está fijado
        if self.ax.get_autoscaley_on():
            all_y = []
            for ydata in data.values():
                all_y.extend(ydata)
            if all_y:
                lo, hi = min(all_y), max(all_y)
                if lo == hi:
                    lo, hi = lo - 1, hi + 1
                pad = (hi - lo) * 0.1
                self.ax.set_ylim(lo - pad, hi + pad)
        self.draw_idle()


# ----------------------------------------------------------------------------
class ParamSpin(QDoubleSpinBox):
    """SpinBox con tooltip y configuración compacta."""
    def __init__(self, vmin, vmax, step, decimals, value, suffix='', tip=''):
        super().__init__()
        self.setRange(vmin, vmax)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setValue(value)
        if suffix:
            self.setSuffix(' ' + suffix)
        if tip:
            self.setToolTip(tip)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(110)


class LabeledSpin(QWidget):
    """Etiqueta + spin en una fila compacta."""
    valueChanged = pyqtSignal(float)
    def __init__(self, label, vmin, vmax, step, decimals, value, suffix='', tip=''):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#a1a1aa; font-size:11px;")
        lbl.setMinimumWidth(115)
        if tip:
            lbl.setToolTip(tip)
        self.spin = ParamSpin(vmin, vmax, step, decimals, value, suffix, tip)
        self.spin.valueChanged.connect(self.valueChanged.emit)
        lay.addWidget(lbl)
        lay.addWidget(self.spin, 1)

    def value(self): return self.spin.value()
    def setValue(self, v): self.spin.setValue(v)


# ============================================================================
# MAIN WINDOW
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemelo Digital · Seguidor de Línea PID")
        self.resize(1500, 900)

        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.track = Track('complex')
        sx, sy, st = Track.START_POSE['complex']
        self.config['start'] = {'x': sx, 'y': sy, 'theta': st}
        self.simulator = Simulator(self.config, self.track)

        self._build_ui()
        self._apply_dark_theme()
        self._connect_signals()
        self._populate_widgets()

        self.track_view.set_track(self.track)
        self.track_view.set_simulator(self.simulator)

        # Telemetría buffer para gráficas (deque limitado)
        self.tel_t = deque(maxlen=600)
        self.tel_err = deque(maxlen=600)
        self.tel_pwml = deque(maxlen=600)
        self.tel_pwmr = deque(maxlen=600)
        self.tel_p = deque(maxlen=600)
        self.tel_i = deque(maxlen=600)
        self.tel_d = deque(maxlen=600)

        # Timers
        self.last_step_t = time.perf_counter()
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._on_sim_tick)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)  # 10 Hz para gráficas/HUD pesado
        self.ui_timer.timeout.connect(self._update_charts)
        self.ui_timer.start()

        self.repaint_timer = QTimer(self)
        self.repaint_timer.setInterval(33)  # ~30 fps repintado canvas
        self.repaint_timer.timeout.connect(self.track_view.update)
        self.repaint_timer.start()

        self.running = False
        self.statusBar().showMessage("Listo. Pulsa ▶ Run para iniciar.")

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setStatusBar(QStatusBar())
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ======= panel izquierdo: configuración =======
        self.left_panel = self._build_config_panel()
        scroll = QScrollArea()
        scroll.setWidget(self.left_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)

        # ======= panel central: pista + métricas + charts =======
        center = QWidget()
        c_lay = QVBoxLayout(center)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(6)

        # toolbar superior
        c_lay.addLayout(self._build_toolbar())

        # vista de pista
        self.track_view = TrackView()
        self.track_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        c_lay.addWidget(self.track_view, 1)

        # métricas
        self.metrics_box = self._build_metrics_box()
        c_lay.addWidget(self.metrics_box)

        # gráficas en tabs para no saturar verticalmente
        chart_tabs = QTabWidget()
        chart_tabs.setMaximumHeight(220)
        self.chart_err = LiveChart("Error  [-3000..3000]",
                                   [("error", "#22d3ee")])
        self.chart_pwm = LiveChart("PWM motores  [0..1]",
                                   [("L", "#a3e635"), ("R", "#f472b6")],
                                   ylim=(-0.05, 1.05))
        self.chart_pid = LiveChart("Componentes PID",
                                   [("P", "#22d3ee"),
                                    ("I", "#fbbf24"),
                                    ("D", "#e879f9")])
        chart_tabs.addTab(self.chart_err, "Error")
        chart_tabs.addTab(self.chart_pwm, "PWM")
        chart_tabs.addTab(self.chart_pid, "PID")
        c_lay.addWidget(chart_tabs)

        outer.addWidget(scroll)
        outer.addWidget(center, 1)

        # ======= panel derecho: zona muerta + estado =======
        right = self._build_right_panel()
        outer.addWidget(right)

    # ------------------------------------------------------------------
    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(260)
        v = QVBoxLayout(panel)
        v.setSpacing(8)
        v.setContentsMargins(4, 4, 4, 4)

        # ── ZONA MUERTA ──────────────────────────────────────────────────
        g_dz = QGroupBox("ZONA MUERTA")
        g_dz_lay = QVBoxLayout(g_dz)
        g_dz_lay.setSpacing(8)

        # Enable toggle
        from PyQt5.QtWidgets import QCheckBox
        self.chk_dz = QCheckBox("Activar zona muerta")
        self.chk_dz.setStyleSheet("color:#e4e4e7; font-size:11px;")
        self.chk_dz.setToolTip(
            "Cuando el error está dentro de la zona muerta,\n"
            "la corrección PID se fuerza a cero."
        )
        g_dz_lay.addWidget(self.chk_dz)

        # Spinner valor
        self.in_dz = LabeledSpin(
            "Umbral ±", 0.0, 3000.0, 10.0, 1,
            0.0, '',
            "Valor calculado por el modelo ML.\n"
            "Si abs(error) < umbral → corrección = 0"
        )
        g_dz_lay.addWidget(self.in_dz)

        # Referencia percentiles
        ref_lbl = QLabel("Referencia rápida:")
        ref_lbl.setStyleSheet("color:#71717a;font-size:10px;margin-top:4px;")
        g_dz_lay.addWidget(ref_lbl)

        self.btn_dz_80 = QPushButton("Conservadora 80%")
        self.btn_dz_90 = QPushButton("Recomendada 90% ★")
        self.btn_dz_95 = QPushButton("Agresiva 95%")
        for b in (self.btn_dz_80, self.btn_dz_90, self.btn_dz_95):
            b.setMinimumHeight(26)
            b.setStyleSheet(
                "QPushButton{background:#18181b;color:#a1a1aa;"
                "border:1px solid #3f3f46;font-size:10px;padding:3px;}"
                "QPushButton:hover{background:#27272a;color:#e4e4e7;}"
            )
            g_dz_lay.addWidget(b)

        # Indicador activo
        self.lbl_dz_status = QLabel("⬤  INACTIVA")
        self.lbl_dz_status.setAlignment(Qt.AlignCenter)
        self.lbl_dz_status.setStyleSheet(
            "color:#71717a;font-family:monospace;"
            "font-size:12px;font-weight:bold;padding:4px;"
            "border:1px solid #27272a;background:#0a0a0b;"
        )
        g_dz_lay.addWidget(self.lbl_dz_status)

        # Indicador "corrección bloqueada"
        self.lbl_dz_block = QLabel("")
        self.lbl_dz_block.setAlignment(Qt.AlignCenter)
        self.lbl_dz_block.setStyleSheet(
            "color:#fbbf24;font-family:monospace;font-size:11px;"
        )
        g_dz_lay.addWidget(self.lbl_dz_block)

        v.addWidget(g_dz)

        # ── ESTADO DEL ROBOT ─────────────────────────────────────────────
        g_est = QGroupBox("ESTADO DEL ROBOT")
        g_est_lay = QVBoxLayout(g_est)
        g_est_lay.setSpacing(6)

        self.lbl_estado_big = QLabel("CENTRADO")
        self.lbl_estado_big.setAlignment(Qt.AlignCenter)
        self.lbl_estado_big.setStyleSheet(
            "color:#a3e635;font-family:monospace;font-size:18px;"
            "font-weight:bold;padding:10px;"
            "border:2px solid #a3e635;background:#0a0a0b;"
        )
        g_est_lay.addWidget(self.lbl_estado_big)

        # Leyenda de colores
        leyenda = [
            ("centrado",      "#a3e635", "Error < 300 · en línea"),
            ("desviado",      "#f87171", "Error aumentando"),
            ("recuperacion",  "#fbbf24", "Error disminuyendo"),
            ("linea_perdida", "#ef4444", "Sin señal en sensores"),
        ]
        for estado, color, desc in leyenda:
            row = QHBoxLayout()
            dot = QLabel("⬤")
            dot.setStyleSheet(f"color:{color};font-size:10px;")
            dot.setFixedWidth(16)
            txt = QLabel(f"<b>{estado}</b><br><span style='color:#71717a;font-size:9px;'>{desc}</span>")
            txt.setStyleSheet("color:#a1a1aa;font-size:10px;")
            txt.setWordWrap(True)
            row.addWidget(dot)
            row.addWidget(txt, 1)
            g_est_lay.addLayout(row)

        v.addWidget(g_est)
        v.addStretch()
        return panel

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setObjectName("runBtn")
        self.btn_run.setToolTip("Iniciar / pausar simulación (Espacio)")
        self.btn_run.setShortcut("Space")
        self.btn_run.setMinimumHeight(32)

        self.btn_reset = QPushButton("⟲  Reset")
        self.btn_reset.setToolTip("Reiniciar al estado inicial")
        self.btn_reset.setMinimumHeight(32)

        self.btn_place = QPushButton("⊕  Colocar robot")
        self.btn_place.setCheckable(True)
        self.btn_place.setToolTip("Click sobre la pista para colocar el robot")
        self.btn_place.setMinimumHeight(32)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#3f3f46;")

        bar.addWidget(self.btn_run)
        bar.addWidget(self.btn_reset)
        bar.addWidget(self.btn_place)
        bar.addWidget(sep)

        bar.addWidget(QLabel("Velocidad sim:"))
        self.sim_speed_slider = QSlider(Qt.Horizontal)
        self.sim_speed_slider.setRange(1, 50)  # 0.1x ... 5.0x
        self.sim_speed_slider.setValue(10)
        self.sim_speed_slider.setMaximumWidth(140)
        self.sim_speed_lbl = QLabel("1.0x")
        self.sim_speed_lbl.setMinimumWidth(35)
        self.sim_speed_lbl.setStyleSheet("color:#22d3ee;font-family:monospace;")
        bar.addWidget(self.sim_speed_slider)
        bar.addWidget(self.sim_speed_lbl)

        bar.addStretch()

        self.btn_export_csv = QPushButton("⇩ Telemetría CSV")
        self.btn_export_csv.setToolTip("Exportar la telemetría completa a CSV")
        self.btn_load_preset = QPushButton("⇧ Preset")
        self.btn_save_preset = QPushButton("⇩ Preset")
        bar.addWidget(self.btn_load_preset)
        bar.addWidget(self.btn_save_preset)
        bar.addWidget(self.btn_export_csv)

        return bar

    # ------------------------------------------------------------------
    def _build_metrics_box(self):
        box = QFrame()
        box.setStyleSheet("background:#18181b;border:1px solid #27272a;")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(20)

        self.metric_labels = {}
        names = [
            ("tiempo_s", "Tiempo", "s", "#22d3ee"),
            ("distancia_m", "Distancia", "m", "#22d3ee"),
            ("vel_media_mps", "V. media", "m/s", "#a3e635"),
            ("rms_error", "RMS error", "", "#fbbf24"),
            ("max_error", "Max error", "", "#fbbf24"),
            ("oscilaciones", "Oscil.", "", "#e879f9"),
            ("porc_en_linea", "En línea", "%", "#a3e635"),
        ]
        for key, label, unit, color in names:
            cell = QVBoxLayout()
            cell.setSpacing(0)
            l1 = QLabel(label.upper())
            l1.setStyleSheet("color:#71717a;font-size:9px;letter-spacing:1px;")
            l2 = QLabel("0")
            l2.setStyleSheet(f"color:{color};font-family:monospace;font-size:14px;")
            cell.addWidget(l1)
            cell.addWidget(l2)
            lay.addLayout(cell)
            self.metric_labels[key] = (l2, unit)
        lay.addStretch()
        return box

    # ------------------------------------------------------------------
    def _build_config_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setSpacing(6)
        v.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.addTab(self._tab_robot(), "Robot")
        tabs.addTab(self._tab_motor_sensor(), "Motor / Sensor")
        tabs.addTab(self._tab_pid(), "PID")
        tabs.addTab(self._tab_track(), "Pista")
        v.addWidget(tabs)
        return panel

    def _tab_robot(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)

        g1 = QGroupBox("Chasis")
        g1l = QVBoxLayout(g1)
        self.in_mass = LabeledSpin("Masa", 0.05, 5.0, 0.01, 3,
                                   self.config['robot']['mass'], 'kg',
                                   "Masa total del vehículo (kg)")
        self.in_wb = LabeledSpin("Distancia ejes", 0.04, 0.40, 0.005, 3,
                                 self.config['robot']['wheelBase'], 'm',
                                 "Distancia entre las dos ruedas motrices")
        self.in_chL = LabeledSpin("Largo chasis", 0.05, 0.40, 0.005, 3,
                                  self.config['robot']['chassisLength'], 'm')
        self.in_chW = LabeledSpin("Ancho chasis", 0.04, 0.40, 0.005, 3,
                                  self.config['robot']['chassisWidth'], 'm')
        self.in_wD = LabeledSpin("Diám. rueda", 0.020, 0.150, 0.001, 3,
                                 self.config['robot']['wheelDiameter'], 'm',
                                 "Diámetro de las ruedas (TT estándar = 0.067 m)")
        for w_ in (self.in_mass, self.in_wb, self.in_chL, self.in_chW, self.in_wD):
            g1l.addWidget(w_)

        g2 = QGroupBox("Sensores · layout")
        g2l = QVBoxLayout(g2)
        self.in_sp = LabeledSpin("Espaciado entre", 0.004, 0.05, 0.0005, 4,
                                 self.config['robot']['sensorSpacing'], 'm',
                                 "Distancia entre sensores adyacentes")
        self.in_sd = LabeledSpin("Dist. al eje", 0.02, 0.20, 0.005, 3,
                                 self.config['robot']['sensorAxisDistance'], 'm',
                                 "Distancia paralela del eje motores a la fila de sensores")
        g2l.addWidget(self.in_sp)
        g2l.addWidget(self.in_sd)

        g3 = QGroupBox("Fricción")
        g3l = QVBoxLayout(g3)
        self.in_rf = LabeledSpin("Coef. rodante μ", 0.0, 0.30, 0.005, 3,
                                 self.config['robot']['rollingFriction'], '',
                                 "Coeficiente de fricción rodante (longitudinal)")
        self.in_lf = LabeledSpin("Lateral b", 0.0, 50.0, 0.5, 2,
                                 self.config['robot']['lateralFriction'], 'N·s/m',
                                 "Arrastre lateral (resistencia a deslizar de costado)")
        g3l.addWidget(self.in_rf)
        g3l.addWidget(self.in_lf)

        g4 = QGroupBox("Pose inicial")
        g4l = QVBoxLayout(g4)
        self.in_x0 = LabeledSpin("X start", 0.0, 5.0, 0.01, 3,
                                 self.config['start']['x'], 'm')
        self.in_y0 = LabeledSpin("Y start", 0.0, 5.0, 0.01, 3,
                                 self.config['start']['y'], 'm')
        self.in_th0 = LabeledSpin("θ start", -np.pi, np.pi, 0.05, 3,
                                  self.config['start']['theta'], 'rad')
        for w_ in (self.in_x0, self.in_y0, self.in_th0):
            g4l.addWidget(w_)

        v.addWidget(g1)
        v.addWidget(g2)
        v.addWidget(g3)
        v.addWidget(g4)
        v.addStretch()
        return w

    def _tab_motor_sensor(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g1 = QGroupBox("Motor TT 48:1")
        g1l = QVBoxLayout(g1)
        self.in_volt = LabeledSpin("Voltaje suministro", 3.0, 24.0, 0.1, 2,
                                   self.config['motor']['voltage'], 'V',
                                   "Tensión real de alimentación")
        self.in_vnom = LabeledSpin("V nominal", 3.0, 24.0, 0.1, 2,
                                   self.config['motor']['nominalVoltage'], 'V',
                                   "Voltaje al que se especifican RPM y torque")
        self.in_rpm = LabeledSpin("RPM no carga", 50, 2000, 10, 0,
                                  self.config['motor']['noLoadRPM_at_nominal'], 'rpm',
                                  "Velocidad sin carga al voltaje nominal")
        self.in_tau = LabeledSpin("Torque stall", 0.01, 1.0, 0.005, 4,
                                  self.config['motor']['stallTorque'], 'N·m',
                                  "Torque máximo a 0 rpm")
        for w_ in (self.in_volt, self.in_vnom, self.in_rpm, self.in_tau):
            g1l.addWidget(w_)

        g2 = QGroupBox("Sensores QTR · 4 ch")
        g2l = QVBoxLayout(g2)
        self.in_fs = LabeledSpin("Frecuencia muestreo", 20, 2000, 10, 0,
                                 self.config['sensor']['samplingRate'], 'Hz',
                                 "Frecuencia de lectura de sensores y PID")
        self.in_sr = LabeledSpin("Apertura óptica", 0.0005, 0.01, 0.0005, 4,
                                 self.config['sensor']['sensorRadius'], 'm',
                                 "Radio de la zona iluminada por cada LED")
        self.in_np = LabeledSpin("Pts. promedio", 1, 25, 1, 0,
                                 self.config['sensor']['samplingPoints'], '',
                                 "Más puntos = más penumbra y suavidad")
        for w_ in (self.in_fs, self.in_sr, self.in_np):
            g2l.addWidget(w_)

        # lecturas en vivo
        g3 = QGroupBox("Lecturas en vivo (0..1023)")
        self.sensor_bars_lay = QGridLayout(g3)
        self.sensor_value_labels = []
        self.sensor_bar_labels = []
        for i in range(4):
            l = QLabel(f"S{i}")
            l.setStyleSheet("color:#71717a;font-size:10px;")
            v_lbl = QLabel("0")
            v_lbl.setStyleSheet("color:#22d3ee;font-family:monospace;font-size:12px;")
            v_lbl.setAlignment(Qt.AlignCenter)
            bar = QFrame()
            bar.setFixedHeight(8)
            bar.setStyleSheet("background:#22d3ee;")
            bar_bg = QFrame()
            bar_bg.setStyleSheet("background:#27272a;")
            bar_lay = QHBoxLayout(bar_bg)
            bar_lay.setContentsMargins(0, 0, 0, 0)
            bar_lay.addWidget(bar)
            bar_bg.setFixedHeight(8)
            self.sensor_bars_lay.addWidget(l,    0, i, alignment=Qt.AlignCenter)
            self.sensor_bars_lay.addWidget(v_lbl, 1, i, alignment=Qt.AlignCenter)
            self.sensor_bars_lay.addWidget(bar_bg, 2, i)
            self.sensor_value_labels.append(v_lbl)
            self.sensor_bar_labels.append((bar, bar_bg))

        v.addWidget(g1)
        v.addWidget(g2)
        v.addWidget(g3)
        v.addStretch()
        return w

    def _tab_pid(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g1 = QGroupBox("Ganancias PID")
        g1l = QVBoxLayout(g1)
        self.in_kp = LabeledSpin("Kp", 0.0, 1.0, 0.0001, 6,
                                 self.config['pid']['kp'], '',
                                 "Ganancia proporcional")
        self.in_ki = LabeledSpin("Ki", 0.0, 1.0, 0.0001, 6,
                                 self.config['pid']['ki'], '',
                                 "Ganancia integral (cero por defecto)")
        self.in_kd = LabeledSpin("Kd", 0.0, 1.0, 0.0001, 6,
                                 self.config['pid']['kd'], '',
                                 "Ganancia derivativa")
        self.in_df = LabeledSpin("Filtro D (EMA)", 0.0, 0.99, 0.05, 2,
                                 self.config['pid']['derivativeFilter'], '',
                                 "0=sin filtro, 0.99=muy filtrado")
        self.in_il = LabeledSpin("Anti-windup ±", 50, 5000, 50, 0,
                                 self.config['pid']['integralLimit'], '',
                                 "Saturación del integrador")
        for w_ in (self.in_kp, self.in_ki, self.in_kd, self.in_df, self.in_il):
            g1l.addWidget(w_)

        g2 = QGroupBox("Velocidad motores (PWM 0..1)")
        g2l = QVBoxLayout(g2)
        self.in_base = LabeledSpin("Velocidad base", 0.0, 1.0, 0.01, 3,
                                   self.config['pid']['baseSpeed'], '',
                                   "PWM nominal cuando error = 0")
        self.in_max = LabeledSpin("Velocidad máx", 0.0, 1.0, 0.01, 3,
                                  self.config['pid']['maxSpeed'], '',
                                  "Tope absoluto del PWM")
        g2l.addWidget(self.in_base)
        g2l.addWidget(self.in_max)

        # salidas en vivo
        g3 = QGroupBox("Salida actual")
        g3l = QGridLayout(g3)
        self.lbl_p = QLabel("0")
        self.lbl_i = QLabel("0")
        self.lbl_d = QLabel("0")
        for i, (lbl, txt, color) in enumerate([(self.lbl_p, "P", "#22d3ee"),
                                                (self.lbl_i, "I", "#fbbf24"),
                                                (self.lbl_d, "D", "#e879f9")]):
            t = QLabel(txt)
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet(f"color:{color};font-size:11px;font-weight:bold;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#e4e4e7;font-family:monospace;font-size:12px;")
            g3l.addWidget(t, 0, i)
            g3l.addWidget(lbl, 1, i)

        v.addWidget(g1)
        v.addWidget(g2)
        v.addWidget(g3)
        v.addStretch()
        return w

    def _tab_track(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g1 = QGroupBox("Generador")
        g1l = QGridLayout(g1)
        self.btns_track = {}
        labels = [('oval', 'Óvalo'), ('square', 'Cuadrado'),
                  ('figure8', 'Figura 8'), ('scurve', 'Curva S'),
                  ('race', 'Race'), ('complex', 'Cruces')]
        for i, (kid, lbl) in enumerate(labels):
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.setMinimumHeight(28)
            b.clicked.connect(lambda _, k=kid: self._load_generated(k))
            g1l.addWidget(b, i // 2, i % 2)
            self.btns_track[kid] = b
        self.btns_track['complex'].setChecked(True)

        g2 = QGroupBox("Cargar PNG")
        g2l = QVBoxLayout(g2)
        info = QLabel("Fondo blanco, líneas negras 1.9 cm.\nEscala: 10 px = 1 cm.")
        info.setStyleSheet("color:#71717a;font-size:11px;")
        info.setWordWrap(True)
        self.btn_load_png = QPushButton("Seleccionar archivo…")
        self.btn_load_png.setMinimumHeight(28)
        g2l.addWidget(info)
        g2l.addWidget(self.btn_load_png)

        g3 = QGroupBox("Información")
        g3l = QVBoxLayout(g3)
        self.lbl_track_info = QLabel("")
        self.lbl_track_info.setStyleSheet("color:#a1a1aa;font-family:monospace;font-size:11px;")
        g3l.addWidget(self.lbl_track_info)

        v.addWidget(g1)
        v.addWidget(g2)
        v.addWidget(g3)
        v.addStretch()
        return w

    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_place.toggled.connect(lambda on: self.track_view.set_placing(on))
        self.track_view.robotPlaced.connect(self._on_place_robot)

        self.sim_speed_slider.valueChanged.connect(self._on_sim_speed)

        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_save_preset.clicked.connect(self._save_preset)
        self.btn_load_preset.clicked.connect(self._load_preset)
        self.btn_load_png.clicked.connect(self._load_png)

        # Zona muerta
        self.chk_dz.toggled.connect(self._sync_dead_zone)
        self.in_dz.valueChanged.connect(self._sync_dead_zone)
        self.btn_dz_80.clicked.connect(lambda: self.in_dz.setValue(240.0))
        self.btn_dz_90.clicked.connect(lambda: self.in_dz.setValue(300.0))
        self.btn_dz_95.clicked.connect(lambda: self.in_dz.setValue(380.0))

        # Conectar TODOS los inputs a updater
        for w in self._iter_inputs():
            w.valueChanged.connect(self._sync_config)

    def _iter_inputs(self):
        return [
            self.in_mass, self.in_wb, self.in_chL, self.in_chW, self.in_wD,
            self.in_sp, self.in_sd, self.in_rf, self.in_lf,
            self.in_x0, self.in_y0, self.in_th0,
            self.in_volt, self.in_vnom, self.in_rpm, self.in_tau,
            self.in_fs, self.in_sr, self.in_np,
            self.in_kp, self.in_ki, self.in_kd, self.in_df, self.in_il,
            self.in_base, self.in_max,
        ]

    # ------------------------------------------------------------------
    def _populate_widgets(self):
        c = self.config
        self.in_mass.setValue(c['robot']['mass'])
        self.in_wb.setValue(c['robot']['wheelBase'])
        self.in_chL.setValue(c['robot']['chassisLength'])
        self.in_chW.setValue(c['robot']['chassisWidth'])
        self.in_wD.setValue(c['robot']['wheelDiameter'])
        self.in_sp.setValue(c['robot']['sensorSpacing'])
        self.in_sd.setValue(c['robot']['sensorAxisDistance'])
        self.in_rf.setValue(c['robot']['rollingFriction'])
        self.in_lf.setValue(c['robot']['lateralFriction'])
        self.in_x0.setValue(c['start']['x'])
        self.in_y0.setValue(c['start']['y'])
        self.in_th0.setValue(c['start']['theta'])
        self.in_volt.setValue(c['motor']['voltage'])
        self.in_vnom.setValue(c['motor']['nominalVoltage'])
        self.in_rpm.setValue(c['motor']['noLoadRPM_at_nominal'])
        self.in_tau.setValue(c['motor']['stallTorque'])
        self.in_fs.setValue(c['sensor']['samplingRate'])
        self.in_sr.setValue(c['sensor']['sensorRadius'])
        self.in_np.setValue(c['sensor']['samplingPoints'])
        self.in_kp.setValue(c['pid']['kp'])
        self.in_ki.setValue(c['pid']['ki'])
        self.in_kd.setValue(c['pid']['kd'])
        self.in_df.setValue(c['pid']['derivativeFilter'])
        self.in_il.setValue(c['pid']['integralLimit'])
        self.in_base.setValue(c['pid']['baseSpeed'])
        self.in_max.setValue(c['pid']['maxSpeed'])
        self._update_track_info()

    def _sync_config(self):
        c = self.config
        c['robot']['mass'] = self.in_mass.value()
        c['robot']['wheelBase'] = self.in_wb.value()
        c['robot']['chassisLength'] = self.in_chL.value()
        c['robot']['chassisWidth'] = self.in_chW.value()
        c['robot']['wheelDiameter'] = self.in_wD.value()
        c['robot']['sensorSpacing'] = self.in_sp.value()
        c['robot']['sensorAxisDistance'] = self.in_sd.value()
        c['robot']['rollingFriction'] = self.in_rf.value()
        c['robot']['lateralFriction'] = self.in_lf.value()
        c['start']['x'] = self.in_x0.value()
        c['start']['y'] = self.in_y0.value()
        c['start']['theta'] = self.in_th0.value()
        c['motor']['voltage'] = self.in_volt.value()
        c['motor']['nominalVoltage'] = self.in_vnom.value()
        c['motor']['noLoadRPM_at_nominal'] = self.in_rpm.value()
        c['motor']['stallTorque'] = self.in_tau.value()
        c['sensor']['samplingRate'] = int(self.in_fs.value())
        c['sensor']['sensorRadius'] = self.in_sr.value()
        c['sensor']['samplingPoints'] = int(self.in_np.value())
        c['pid']['kp'] = self.in_kp.value()
        c['pid']['ki'] = self.in_ki.value()
        c['pid']['kd'] = self.in_kd.value()
        c['pid']['derivativeFilter'] = self.in_df.value()
        c['pid']['integralLimit'] = self.in_il.value()
        c['pid']['baseSpeed'] = self.in_base.value()
        c['pid']['maxSpeed'] = self.in_max.value()
        self.simulator.update_config(c)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _sync_dead_zone(self, *_):
        enabled = self.chk_dz.isChecked()
        val = self.in_dz.value()
        self.config['pid']['deadZone'] = val
        self.config['pid']['deadZoneEnabled'] = enabled
        self.simulator.update_config(self.config)
        if enabled:
            self.lbl_dz_status.setText(f"⬤  ACTIVA  ±{val:.0f}")
            self.lbl_dz_status.setStyleSheet(
                "color:#a3e635;font-family:monospace;"
                "font-size:12px;font-weight:bold;padding:4px;"
                "border:1px solid #a3e635;background:#0a0a0b;"
            )
        else:
            self.lbl_dz_status.setText("⬤  INACTIVA")
            self.lbl_dz_status.setStyleSheet(
                "color:#71717a;font-family:monospace;"
                "font-size:12px;font-weight:bold;padding:4px;"
                "border:1px solid #27272a;background:#0a0a0b;"
            )

    def _on_sim_speed(self, v):
        f = v / 10.0
        self.config['sim']['simSpeed'] = f
        self.sim_speed_lbl.setText(f"{f:.1f}x")
        self.simulator.update_config(self.config)

    def _toggle_run(self):
        self.running = not self.running
        if self.running:
            self.btn_run.setText("⏸  Pause")
            self.btn_run.setStyleSheet(self._button_style("amber"))
            self.last_step_t = time.perf_counter()
            self.sim_timer.start(16)  # ~60 Hz
            self.statusBar().showMessage("Simulación corriendo…")
        else:
            self.btn_run.setText("▶  Run")
            self.btn_run.setStyleSheet(self._button_style("green"))
            self.sim_timer.stop()
            self.statusBar().showMessage("Pausado.")

    def _reset(self):
        was = self.running
        if was:
            self._toggle_run()
        # Crear simulator nuevo con config actual
        self.simulator = Simulator(self.config, self.track)
        self.track_view.set_simulator(self.simulator)
        self.tel_t.clear(); self.tel_err.clear()
        self.tel_pwml.clear(); self.tel_pwmr.clear()
        self.tel_p.clear(); self.tel_i.clear(); self.tel_d.clear()
        self._update_charts()
        self.track_view.update()
        self.statusBar().showMessage("Estado reiniciado.")

    def _on_place_robot(self, x_m, y_m):
        self.simulator.x = x_m
        self.simulator.y = y_m
        self.simulator.vx = self.simulator.vy = self.simulator.omega = 0
        self.simulator.trail.clear()
        self.simulator.first_pid = True
        self.simulator.sensors = self.simulator.sense()
        self.in_x0.setValue(x_m)
        self.in_y0.setValue(y_m)
        self.btn_place.setChecked(False)
        self.track_view.set_placing(False)
        self.track_view.update()
        self.statusBar().showMessage(f"Robot en ({x_m*100:.1f}, {y_m*100:.1f}) cm.")

    def _on_sim_tick(self):
        now = time.perf_counter()
        dt = now - self.last_step_t
        self.last_step_t = now
        # Capar dt para evitar grandes saltos si la app se congela
        dt = min(dt, 0.1)
        self.simulator.step(dt)

    # ------------------------------------------------------------------
    def _update_charts(self):
        s = self.simulator
        # Sensor labels
        for i in range(4):
            v = int(s.sensors[i])
            self.sensor_value_labels[i].setText(str(v))
            bar, bg = self.sensor_bar_labels[i]
            total_w = max(1, bg.width())
            bar.setFixedWidth(int(total_w * v / 1023))
        # PID
        self.lbl_p.setText(f"{s.P:+.3f}")
        self.lbl_i.setText(f"{s.I:+.3f}")
        self.lbl_d.setText(f"{s.D:+.3f}")

        # ── Estado del robot (panel derecho) ────────────────────────────
        estado = getattr(s, 'estado', 'centrado')
        estado_styles = {
            'centrado':      ("CENTRADO",      "#a3e635", "#a3e635"),
            'desviado':      ("DESVIADO",      "#f87171", "#f87171"),
            'recuperacion':  ("RECUPERACION",  "#fbbf24", "#fbbf24"),
            'linea_perdida': ("LÍNEA PERDIDA", "#ef4444", "#ef4444"),
        }
        txt, color, border = estado_styles.get(estado, ("CENTRADO", "#a3e635", "#a3e635"))
        self.lbl_estado_big.setText(txt)
        self.lbl_estado_big.setStyleSheet(
            f"color:{color};font-family:monospace;font-size:18px;"
            f"font-weight:bold;padding:10px;"
            f"border:2px solid {border};background:#0a0a0b;"
        )

        # Indicador corrección bloqueada
        if getattr(s, 'corr_blocked', False):
            self.lbl_dz_block.setText("🔒 corrección = 0")
        else:
            self.lbl_dz_block.setText("")

        # Métricas
        m = s.metrics()
        for key, (lbl, unit) in self.metric_labels.items():
            if key in m:
                v = m[key]
                if isinstance(v, float):
                    if key in ('rms_error', 'max_error'):
                        lbl.setText(f"{v:.4f}{unit}")
                    else:
                        lbl.setText(f"{v:.3f}{unit}")
                else:
                    lbl.setText(f"{v}{unit}")
        # Cargar últimos N puntos de history en buffers de gráficas
        # Tomar cada k-ésimo si hay muchos puntos
        if s.history:
            recent = s.history[-600:]
            self.tel_t = [r['t'] for r in recent]
            self.tel_err = [r['error'] for r in recent]
            self.tel_pwml = [r['pwm_l'] for r in recent]
            self.tel_pwmr = [r['pwm_r'] for r in recent]
            self.tel_p = [r['P'] for r in recent]
            self.tel_i = [r['I'] for r in recent]
            self.tel_d = [r['D'] for r in recent]
            self.chart_err.update_series(self.tel_t, {"error": self.tel_err})
            self.chart_pwm.update_series(self.tel_t,
                                         {"L": self.tel_pwml, "R": self.tel_pwmr})
            self.chart_pid.update_series(self.tel_t,
                                         {"P": self.tel_p, "I": self.tel_i, "D": self.tel_d})

    # ------------------------------------------------------------------
    def _load_generated(self, kind):
        for k, b in self.btns_track.items():
            b.setChecked(k == kind)
        self.track = Track(kind)
        self.track_view.set_track(self.track)
        sx, sy, st = Track.START_POSE[kind]
        self.config['start'] = {'x': sx, 'y': sy, 'theta': st}
        self.in_x0.setValue(sx); self.in_y0.setValue(sy); self.in_th0.setValue(st)
        # nuevo simulador
        self.simulator = Simulator(self.config, self.track)
        self.track_view.set_simulator(self.simulator)
        self.tel_t = []; self.tel_err = []
        self._update_track_info()
        self.track_view.update()
        self.statusBar().showMessage(f"Pista '{kind}' cargada.")

    def _load_png(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar pista PNG", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            self.track = Track(path)
            self.track_view.set_track(self.track)
            for b in self.btns_track.values(): b.setChecked(False)
            # poner el robot en el centro de la pista
            cx_m = self.track.w / 2 / PX_PER_M
            cy_m = self.track.h / 2 / PX_PER_M
            self.config['start'] = {'x': cx_m, 'y': cy_m, 'theta': 0.0}
            self.in_x0.setValue(cx_m); self.in_y0.setValue(cy_m); self.in_th0.setValue(0)
            self.simulator = Simulator(self.config, self.track)
            self.track_view.set_simulator(self.simulator)
            self._update_track_info()
            self.track_view.update()
            self.statusBar().showMessage(
                f"PNG cargado. Usa ⊕ Colocar robot para posicionarlo sobre la línea.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar la imagen:\n{e}")

    def _update_track_info(self):
        if self.track:
            self.lbl_track_info.setText(
                f"Origen: {self.track.kind}\n"
                f"Tamaño: {self.track.w} × {self.track.h} px\n"
                f"Equiv:  {self.track.w/10:.1f} × {self.track.h/10:.1f} cm"
            )

    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self.simulator.history:
            QMessageBox.information(self, "Sin datos",
                                    "No hay telemetría todavía. Corre la simulación.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar telemetría",
                                              f"telemetria_{int(time.time())}.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        keys = list(self.simulator.history[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.simulator.history)
        self.statusBar().showMessage(
            f"✓ {len(self.simulator.history)} muestras → {path}")

    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar preset",
                                              "preset.json", "JSON (*.json)")
        if not path:
            return
        with open(path, 'w') as f:
            json.dump({'version': 2, 'config': self.config}, f, indent=2)
        self.statusBar().showMessage(f"✓ Preset guardado: {path}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar preset", "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                obj = json.load(f)
            if 'config' in obj:
                # merge para tolerar presets de versiones distintas
                merged = copy.deepcopy(DEFAULT_CONFIG)
                for k, v in obj['config'].items():
                    if isinstance(v, dict) and k in merged:
                        merged[k].update(v)
                    else:
                        merged[k] = v
                self.config = merged
                self._populate_widgets()
                self._sync_config()
                self.statusBar().showMessage(f"✓ Preset cargado: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar:\n{e}")

    # ------------------------------------------------------------------
    def _button_style(self, accent):
        colors = {
            'green':  ('#a3e635', '#1a2e05'),
            'amber':  ('#fbbf24', '#451a03'),
            'cyan':   ('#22d3ee', '#083344'),
            'default':('#a1a1aa', '#18181b'),
        }
        fg, bg = colors.get(accent, colors['default'])
        return f"""
            QPushButton#runBtn {{
                background:{bg}; color:{fg};
                border:1px solid {fg}; padding:6px 14px;
                font-size:12px; font-weight:bold;
            }}
            QPushButton#runBtn:hover {{ background:{fg}; color:{bg}; }}
        """

    def _apply_dark_theme(self):
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0a0a0b; color:#e4e4e7; }
            QGroupBox {
                border:1px solid #27272a; margin-top:10px; padding-top:8px;
                font-size:11px; color:#a1a1aa; font-weight:bold;
                letter-spacing:1px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left:8px; padding:0 4px;
                background:#0a0a0b;
            }
            QTabWidget::pane { border:1px solid #27272a; background:#18181b; }
            QTabBar::tab {
                background:#18181b; color:#a1a1aa; padding:6px 12px;
                border:1px solid #27272a; font-size:11px;
            }
            QTabBar::tab:selected { background:#0a0a0b; color:#22d3ee;
                border-bottom:2px solid #22d3ee; }
            QPushButton {
                background:#18181b; color:#e4e4e7; border:1px solid #3f3f46;
                padding:5px 10px; font-size:11px;
            }
            QPushButton:hover { background:#27272a; border-color:#52525b; }
            QPushButton:checked {
                background:#083344; color:#22d3ee; border-color:#22d3ee;
            }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background:#18181b; color:#e4e4e7; border:1px solid #3f3f46;
                padding:3px 6px; font-family:monospace; font-size:11px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus { border-color:#22d3ee; }
            QSlider::groove:horizontal {
                background:#27272a; height:4px; border-radius:2px;
            }
            QSlider::handle:horizontal {
                background:#22d3ee; width:14px; margin:-5px 0; border-radius:7px;
            }
            QStatusBar { background:#18181b; color:#a1a1aa; font-size:11px; }
            QToolTip {
                background:#18181b; color:#e4e4e7; border:1px solid #22d3ee;
                padding:4px;
            }
            QScrollArea { border:none; }
            QScrollBar:vertical {
                background:#0a0a0b; width:10px;
            }
            QScrollBar::handle:vertical {
                background:#3f3f46; border-radius:4px; min-height:20px;
            }
            QScrollBar::handle:vertical:hover { background:#52525b; }
        """)
        self.btn_run.setStyleSheet(self._button_style("green"))


# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Sans Serif", 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()