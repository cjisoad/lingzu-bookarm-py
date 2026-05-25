"""RealSense point-cloud picking page for MotorStudio."""

from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from PyQt6.QtCore import QEvent, QSignalBlocker, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from MotorStudio.utils.i18n import tr
from MotorStudio.utils.style import SCENE_COLORS
from MotorStudio.utils.theme_manager import ThemeManager

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkRenderingCore import vtkPointPicker

    HAS_PYVISTA = True
except Exception as _exc:
    pv = None
    QtInteractor = None
    vtkPointPicker = None
    HAS_PYVISTA = False
    logging.getLogger("MotorStudio.realsense_panel").warning(
        "pyvista / pyvistaqt 不可用，RealSense 点云页禁用: %s", _exc
    )


logger = logging.getLogger("MotorStudio.realsense_panel")

M_TO_CM = 100.0


# Camera -> robot base extrinsic:
# camera +X -> robot +Y
# camera +Y -> robot -Z
# camera +Z -> robot -X
# camera origin in robot base: x=+5 cm, y=-28.5 cm, z=+15 cm.
CAMERA_TO_ROBOT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "config"
    / "camera_to_robot_transform.json"
)
DEFAULT_CAMERA_TO_ROBOT_MATRIX = np.array(
    [
        [0.0, 0.0, -1.0, 0.05],
        [1.0, 0.0, 0.0, -0.285],
        [0.0, -1.0, 0.0, 0.15],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _validated_array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _load_camera_to_robot_transform(
    path: Path = CAMERA_TO_ROBOT_CONFIG_PATH,
) -> np.ndarray:
    """Load camera->robot extrinsic from JSON, falling back to built-in values."""

    matrix = DEFAULT_CAMERA_TO_ROBOT_MATRIX.copy()
    if not path.exists():
        return matrix

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")

        if "matrix" in payload:
            matrix = _validated_array(payload["matrix"], (4, 4), "matrix")
        else:
            rotation_values = payload.get(
                "rotation",
                payload.get("rotation_matrix", matrix[:3, :3]),
            )
            rotation = _validated_array(rotation_values, (3, 3), "rotation")

            translation = matrix[:3, 3]
            if "translation_m" in payload:
                translation = _validated_array(
                    payload["translation_m"],
                    (3,),
                    "translation_m",
                )
            elif "translation_cm" in payload:
                translation = (
                    _validated_array(payload["translation_cm"], (3,), "translation_cm")
                    / M_TO_CM
                )
            elif "xyz_m" in payload:
                translation = _validated_array(payload["xyz_m"], (3,), "xyz_m")

            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation

        return matrix.astype(float, copy=True)
    except Exception as exc:
        logger.warning(
            "加载相机到机械臂外参失败，使用内置默认值: %s (%s)",
            path,
            exc,
        )
        return matrix


CAMERA_TO_ROBOT_MATRIX = _load_camera_to_robot_transform()
CAMERA_TO_ROBOT_ROTATION = CAMERA_TO_ROBOT_MATRIX[:3, :3]
CAMERA_TO_ROBOT_TRANSLATION_M = CAMERA_TO_ROBOT_MATRIX[:3, 3]


def _rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cr, sr = math.cos(rx), math.sin(rx)
    cp, sp = math.cos(ry), math.sin(ry)
    cy, sy = math.cos(rz), math.sin(rz)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def camera_point_to_robot_target(
    camera_point_m: Sequence[float],
) -> np.ndarray:
    """Map a RealSense camera-space point into robot base coordinates."""

    camera_point = np.asarray(camera_point_m, dtype=float).reshape(3)
    camera_point_h = np.ones(4, dtype=float)
    camera_point_h[:3] = camera_point
    return (CAMERA_TO_ROBOT_MATRIX @ camera_point_h)[:3]


def apply_tcp_offset_correction(
    robot_point_m: Sequence[float],
    tcp_offset: Sequence[float],
    rpy_rad: Sequence[float],
) -> np.ndarray:
    """Convert a picked TCP point into the underlying reference-frame target."""

    point = np.asarray(robot_point_m, dtype=float).reshape(3)
    offset = np.zeros(6, dtype=float)
    values = [] if tcp_offset is None else list(tcp_offset)
    for idx in range(min(6, len(values))):
        try:
            offset[idx] = float(values[idx])
        except (TypeError, ValueError):
            offset[idx] = 0.0
    rx, ry, rz = np.asarray(rpy_rad, dtype=float).reshape(3)
    tcp_rot = _rpy_to_matrix(offset[3], offset[4], offset[5])
    return point - _rpy_to_matrix(rx, ry, rz) @ tcp_rot.T @ offset[:3]


def _format_vec_m(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3)
    return f"X={vals[0]:.4f} m, Y={vals[1]:.4f} m, Z={vals[2]:.4f} m"


def _format_vec_cm(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3) * M_TO_CM
    return f"X={vals[0]:.2f} cm, Y={vals[1]:.2f} cm, Z={vals[2]:.2f} cm"


def _filter_depth_min(point_cloud, depth_min_m: float):
    if depth_min_m <= 0.0:
        return point_cloud

    points = np.asarray(point_cloud.points_xyz_m)
    keep = points[:, 2] >= depth_min_m
    return type(point_cloud)(
        points_xyz_m=points[keep],
        colors_rgb=(
            None
            if point_cloud.colors_rgb is None
            else np.asarray(point_cloud.colors_rgb)[keep]
        ),
        pixels_uv=np.asarray(point_cloud.pixels_uv)[keep],
        intrinsics=point_cloud.intrinsics,
        timestamp_ms=point_cloud.timestamp_ms,
        frame_number=point_cloud.frame_number,
    )


class RealSenseCaptureWorker(QThread):
    """Capture one RealSense RGB-D frame and convert it to a point cloud."""

    capture_finished = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        warmup: int,
        timeout_ms: int,
        align_depth_to_color: bool,
        depth_min_m: float,
        depth_max_m: float,
        stride: int,
        include_color: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.timeout_ms = timeout_ms
        self.align_depth_to_color = align_depth_to_color
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.stride = stride
        self.include_color = include_color

    def run(self):
        try:
            from el_a3_sdk.realsense import RealSenseD435

            with RealSenseD435(
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=self.serial or None,
                align_depth_to_color=self.align_depth_to_color,
            ) as camera:
                if self.isInterruptionRequested():
                    return
                camera.warmup(frame_count=self.warmup, timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return
                frame = camera.get_frame(timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return

            point_cloud = frame.to_point_cloud(
                max_depth_m=self.depth_max_m,
                stride=self.stride,
                include_color=self.include_color,
            )
            point_cloud = _filter_depth_min(point_cloud, self.depth_min_m)
            if point_cloud.size == 0:
                raise RuntimeError("点云为空，请调整深度范围或相机视角。")
            self.capture_finished.emit(frame, point_cloud)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class RealSensePointPanel(QWidget):
    """Point cloud capture, target picking, and MoveL confirmation page."""

    move_l_requested = pyqtSignal(list, float)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capture_worker: Optional[RealSenseCaptureWorker] = None
        self._frame = None
        self._point_cloud = None
        self._display_indices: Optional[np.ndarray] = None
        self._display_points: Optional[np.ndarray] = None
        self._cloud_actor = None
        self._selected_actor = None
        self._picker = vtkPointPicker() if HAS_PYVISTA else None
        self._filter_installed = False
        self._selected_display_index: Optional[int] = None
        self._selected_display_point_m: Optional[np.ndarray] = None
        self._selected_raw_index: Optional[int] = None
        self._selected_camera_point_m: Optional[np.ndarray] = None
        self._selected_robot_target_raw_m: Optional[np.ndarray] = None
        self._target_robot_point_m: Optional[np.ndarray] = None
        self._tcp_offset = np.zeros(6, dtype=float)
        self._current_end_pose = None
        self._rpy_initialized = False
        self._arm_enabled = False
        self._viewer_visible = False
        self._viewer_widget = None
        self._init_ui()

    def _init_ui(self):
        self._viewer_widget = self._create_viewer_widget()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(self._create_capture_group())
        controls_layout.addWidget(self._create_transform_group())
        controls_layout.addWidget(self._create_result_group())
        controls_layout.addWidget(self._create_move_group())
        controls_layout.addStretch()
        root.addLayout(controls_layout, 1)

        self.status_label = QLabel(tr("pc.ready"))
        root.addWidget(self.status_label)
        self._update_move_button_state()

    def _create_viewer_widget(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAS_PYVISTA:
            self._plotter = None
            self._plotter_placeholder = QLabel(tr("pc.viewer_loading"))
            self._plotter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plotter_placeholder.setWordWrap(True)
            layout.addWidget(self._plotter_placeholder)
        else:
            self._plotter = None
            self.no_plot_label = QLabel(tr("pc.no_pyvista"))
            self.no_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.no_plot_label.setWordWrap(True)
            layout.addWidget(self.no_plot_label)
        return container

    def _ensure_plotter(self):
        if not HAS_PYVISTA or self._viewer_widget is None or self._plotter is not None:
            return

        layout = self._viewer_widget.layout()
        if layout is None:
            layout = QVBoxLayout(self._viewer_widget)
            layout.setContentsMargins(0, 0, 0, 0)

        pv.global_theme.allow_empty_mesh = True
        self._plotter = QtInteractor(self._viewer_widget, multi_samples=4)
        self._plotter.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])

        if hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            layout.removeWidget(self._plotter_placeholder)
            self._plotter_placeholder.deleteLater()
            self._plotter_placeholder = None

        layout.addWidget(self._plotter.interactor)
        self._install_event_filter()
        self._reset_scene()

    def viewer_widget(self):
        return self._viewer_widget

    def show_viewer(self):
        self._viewer_visible = True
        self._ensure_plotter()
        if self._point_cloud is not None and self._display_points is not None:
            self._display_cloud(self._point_cloud)
        else:
            self._reset_scene()
        self._update_pick_mode_state()

    def hide_viewer(self):
        self._viewer_visible = False
        self._update_pick_mode_state()

    def _create_capture_group(self):
        self.capture_group = QGroupBox(tr("pc.capture_group"))
        layout = QGridLayout(self.capture_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.serial_label = QLabel(tr("pc.serial"))
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText(tr("pc.serial_placeholder"))
        layout.addWidget(self.serial_label, 0, 0)
        layout.addWidget(self.serial_edit, 0, 1)

        self.width_label = QLabel(tr("pc.width"))
        self.width_spin = self._make_int_spin(160, 1920, 640, 10)
        layout.addWidget(self.width_label, 0, 2)
        layout.addWidget(self.width_spin, 0, 3)

        self.height_label = QLabel(tr("pc.height"))
        self.height_spin = self._make_int_spin(120, 1080, 480, 10)
        layout.addWidget(self.height_label, 0, 4)
        layout.addWidget(self.height_spin, 0, 5)

        self.fps_label = QLabel(tr("pc.fps"))
        self.fps_spin = self._make_int_spin(1, 90, 30, 1)
        layout.addWidget(self.fps_label, 1, 0)
        layout.addWidget(self.fps_spin, 1, 1)

        self.warmup_label = QLabel(tr("pc.warmup"))
        self.warmup_spin = self._make_int_spin(0, 120, 30, 1)
        layout.addWidget(self.warmup_label, 1, 2)
        layout.addWidget(self.warmup_spin, 1, 3)

        self.timeout_label = QLabel(tr("pc.timeout"))
        self.timeout_spin = self._make_int_spin(500, 30000, 5000, 500)
        self.timeout_spin.setSuffix(" ms")
        layout.addWidget(self.timeout_label, 1, 4)
        layout.addWidget(self.timeout_spin, 1, 5)

        self.depth_min_label = QLabel(tr("pc.depth_min"))
        self.depth_min_spin = self._make_float_spin(0.0, 10.0, 0.0, 0.01, " m")
        layout.addWidget(self.depth_min_label, 2, 0)
        layout.addWidget(self.depth_min_spin, 2, 1)

        self.depth_max_label = QLabel(tr("pc.depth_max"))
        self.depth_max_spin = self._make_float_spin(0.01, 10.0, 2.0, 0.01, " m")
        layout.addWidget(self.depth_max_label, 2, 2)
        layout.addWidget(self.depth_max_spin, 2, 3)

        self.stride_label = QLabel(tr("pc.stride"))
        self.stride_spin = self._make_int_spin(1, 16, 1, 1)
        layout.addWidget(self.stride_label, 2, 4)
        layout.addWidget(self.stride_spin, 2, 5)

        self.max_points_label = QLabel(tr("pc.max_points"))
        self.max_points_spin = self._make_int_spin(1000, 400000, 150000, 10000)
        layout.addWidget(self.max_points_label, 3, 0)
        layout.addWidget(self.max_points_spin, 3, 1)

        self.point_size_label = QLabel(tr("pc.point_size"))
        self.point_size_spin = self._make_float_spin(1.0, 10.0, 2.0, 0.5, "")
        self.point_size_spin.valueChanged.connect(
            lambda _value: self._redisplay_current_cloud()
        )
        layout.addWidget(self.point_size_label, 3, 2)
        layout.addWidget(self.point_size_spin, 3, 3)

        self.align_check = QCheckBox(tr("pc.align_depth"))
        self.align_check.setChecked(True)
        layout.addWidget(self.align_check, 3, 4, 1, 2)

        self.color_check = QCheckBox(tr("pc.use_color"))
        self.color_check.setChecked(True)
        layout.addWidget(self.color_check, 4, 0, 1, 2)

        self.flip_view_check = QCheckBox(tr("pc.flip_view"))
        self.flip_view_check.setChecked(True)
        self.flip_view_check.toggled.connect(
            lambda _checked: self._redisplay_current_cloud()
        )
        layout.addWidget(self.flip_view_check, 4, 2, 1, 2)

        self.pick_mode_btn = QPushButton(tr("pc.pick_mode"))
        self.pick_mode_btn.setCheckable(True)
        self.pick_mode_btn.setChecked(True)
        self.pick_mode_btn.toggled.connect(
            lambda _checked: self._update_pick_mode_state()
        )
        layout.addWidget(self.pick_mode_btn, 4, 4)

        self.capture_btn = QPushButton(tr("pc.capture"))
        self.capture_btn.setObjectName("enableBtn")
        self.capture_btn.clicked.connect(self._start_capture)
        layout.addWidget(self.capture_btn, 4, 5)

        return self.capture_group

    def _create_transform_group(self):
        self.transform_group = QGroupBox(tr("pc.transform_group"))
        layout = QVBoxLayout(self.transform_group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        self.transform_summary = QLabel("")
        self.transform_summary.setWordWrap(True)
        layout.addWidget(self.transform_summary)
        self._update_transform_summary()
        return self.transform_group

    def _create_result_group(self):
        self.result_group = QGroupBox(tr("pc.result_group"))
        layout = QFormLayout(self.result_group)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.pixel_label = QLabel(tr("pc.pixel"))
        self.camera_point_label = QLabel(tr("pc.camera_point"))
        self.robot_point_cm_label = QLabel(tr("pc.move_target_point_cm"))
        self.robot_point_m_label = QLabel(tr("pc.move_target_point_m"))
        self.target_point_cm_label = QLabel(tr("pc.target_point_cm"))
        self.target_point_m_label = QLabel(tr("pc.target_point_m"))

        self.pixel_value = QLabel("--")
        self.camera_point_value = QLabel("--")
        self.move_target_point_cm_value = QLabel("--")
        self.move_target_point_m_value = QLabel("--")
        self.target_point_cm_value = QLabel("--")
        self.target_point_m_value = QLabel("--")

        layout.addRow(self.pixel_label, self.pixel_value)
        layout.addRow(self.camera_point_label, self.camera_point_value)
        layout.addRow(self.robot_point_cm_label, self.move_target_point_cm_value)
        layout.addRow(self.robot_point_m_label, self.move_target_point_m_value)
        layout.addRow(self.target_point_cm_label, self.target_point_cm_value)
        layout.addRow(self.target_point_m_label, self.target_point_m_value)
        return self.result_group

    def _create_move_group(self):
        self.move_group = QGroupBox(tr("pc.move_group"))
        layout = QGridLayout(self.move_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.rpy_labels = []
        self.rpy_spins = []
        for col, key in enumerate(("pc.rx", "pc.ry", "pc.rz")):
            label = QLabel(tr(key))
            spin = self._make_float_spin(-180.0, 180.0, 0.0, 1.0, "°")
            spin.valueChanged.connect(self._on_rpy_changed)
            self.rpy_labels.append(label)
            self.rpy_spins.append(spin)
            layout.addWidget(label, 0, col * 2)
            layout.addWidget(spin, 0, col * 2 + 1)

        self.duration_label = QLabel(tr("pc.duration"))
        self.duration_spin = self._make_float_spin(0.5, 30.0, 2.0, 0.5, " s")
        layout.addWidget(self.duration_label, 1, 0)
        layout.addWidget(self.duration_spin, 1, 1)

        row = QHBoxLayout()
        self.read_rpy_btn = QPushButton(tr("pc.read_rpy"))
        self.read_rpy_btn.clicked.connect(self._fill_current_rpy)
        row.addWidget(self.read_rpy_btn)

        self.move_btn = QPushButton(tr("pc.confirm_move"))
        self.move_btn.setObjectName("enableBtn")
        self.move_btn.clicked.connect(self._on_confirm_move)
        row.addWidget(self.move_btn)
        row.addStretch()
        layout.addLayout(row, 1, 2, 1, 4)
        return self.move_group

    @staticmethod
    def _make_int_spin(lo: int, hi: int, value: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedWidth(105)
        return spin

    @staticmethod
    def _make_float_spin(
        lo: float,
        hi: float,
        value: float,
        step: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3 if abs(step) < 0.1 else 2)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setFixedWidth(115)
        return spin

    def _start_capture(self):
        if self._capture_worker is not None and self._capture_worker.isRunning():
            return
        if self.depth_max_spin.value() <= self.depth_min_spin.value():
            self._set_error(tr("pc.depth_range_error"))
            return

        self.capture_btn.setEnabled(False)
        self.capture_btn.setText(tr("pc.capturing"))
        self.status_label.setText(tr("pc.capturing"))
        self._clear_selection()

        self._capture_worker = RealSenseCaptureWorker(
            serial=self.serial_edit.text().strip() or None,
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            fps=self.fps_spin.value(),
            warmup=self.warmup_spin.value(),
            timeout_ms=self.timeout_spin.value(),
            align_depth_to_color=self.align_check.isChecked(),
            depth_min_m=self.depth_min_spin.value(),
            depth_max_m=self.depth_max_spin.value(),
            stride=self.stride_spin.value(),
            include_color=self.color_check.isChecked(),
            parent=self,
        )
        self._capture_worker.capture_finished.connect(self._on_capture_finished)
        self._capture_worker.error_occurred.connect(self._on_capture_error)
        self._capture_worker.finished.connect(self._on_capture_thread_finished)
        self._capture_worker.start()

    def _on_capture_finished(self, frame, point_cloud):
        self._frame = frame
        self._point_cloud = point_cloud
        self._selected_display_index = None
        self._selected_raw_index = None
        self._selected_camera_point_m = None
        self._selected_robot_target_raw_m = None
        self._target_robot_point_m = None
        self._display_cloud(point_cloud)
        self.pick_mode_btn.setChecked(True)
        self.status_label.setText(
            tr("pc.capture_done", n=point_cloud.size, frame=frame.frame_number)
        )
        self.log_message.emit(
            tr("pc.capture_done", n=point_cloud.size, frame=frame.frame_number)
        )
        self._update_move_button_state()

    def _on_capture_error(self, message: str):
        self._set_error(tr("pc.capture_error", msg=message))

    def _on_capture_thread_finished(self):
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText(tr("pc.capture"))

    def _display_cloud(self, point_cloud):
        self._ensure_plotter()
        if not HAS_PYVISTA or self._plotter is None:
            return

        points = np.asarray(point_cloud.points_xyz_m, dtype=np.float64)
        count = len(points)
        max_points = max(1, self.max_points_spin.value())
        if count > max_points:
            indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
        else:
            indices = np.arange(count, dtype=np.int64)

        display_points = points[indices].copy()
        if self.flip_view_check.isChecked():
            display_points[:, 1] *= -1.0
            display_points[:, 2] *= -1.0

        self._display_indices = indices
        self._display_points = display_points
        colors = None
        if point_cloud.colors_rgb is not None:
            colors = np.asarray(point_cloud.colors_rgb, dtype=np.uint8)[indices]

        self._reset_scene()
        cloud_poly = pv.PolyData(display_points)
        cloud_poly["point_id"] = np.arange(len(display_points), dtype=np.int32)
        cloud_poly.verts = np.column_stack(
            (
                np.ones(len(display_points), dtype=np.int64),
                np.arange(len(display_points), dtype=np.int64),
            )
        ).ravel()
        point_size = float(self.point_size_spin.value())
        if colors is not None and len(colors) == len(display_points):
            cloud_poly["rgb"] = colors
            self._cloud_actor = self._plotter.add_mesh(
                cloud_poly,
                scalars="rgb",
                rgb=True,
                point_size=point_size,
                render_points_as_spheres=True,
                name="realsense_cloud",
            )
        else:
            self._cloud_actor = self._plotter.add_mesh(
                cloud_poly,
                color="#8bd5ff",
                point_size=point_size,
                render_points_as_spheres=True,
                name="realsense_cloud",
            )

        if self._picker is not None and self._cloud_actor is not None:
            try:
                self._picker.InitializePickList()
                self._picker.AddPickList(self._cloud_actor)
                self._picker.PickFromListOn()
                self._picker.SetTolerance(0.01)
            except Exception:
                pass

        try:
            self._plotter.reset_camera()
            self._apply_initial_view()
            if (
                self._selected_display_index is not None
                and self._selected_display_index < len(self._display_points)
            ):
                self._add_selection_marker(
                    self._selected_display_index,
                    self._display_points[self._selected_display_index],
                )
            self._plotter.render()
        except Exception:
            pass

    def _redisplay_current_cloud(self):
        if self._point_cloud is not None:
            self._display_cloud(self._point_cloud)
        self._update_pick_mode_state()

    def _reset_scene(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        try:
            self._plotter.clear()
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])
            self._plotter.add_axes()
        except Exception:
            pass
        self._cloud_actor = None
        self._selected_actor = None

    def _install_event_filter(self):
        if not HAS_PYVISTA or self._plotter is None or self._filter_installed:
            return
        self._plotter.interactor.installEventFilter(self)
        self._filter_installed = True

    def eventFilter(self, obj, event):
        if (
            not HAS_PYVISTA
            or self._plotter is None
            or obj is not self._plotter.interactor
            or not hasattr(self, "pick_mode_btn")
            or not self.pick_mode_btn.isChecked()
        ):
            return False

        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and self._display_points is not None
        ):
            picked = self._pick_display_point(event)
            if picked is not None:
                display_index, display_point = picked
                self._select_display_point(display_index, display_point)
                return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            return False
        return False

    def _pick_display_point(self, event) -> Optional[tuple[int, np.ndarray]]:
        if self._display_points is None or self._plotter is None:
            return None

        vx, vy = self._qt_to_vtk_coords(event)
        picked = self._pick_nearest_screen_point(vx, vy)
        if picked is not None:
            return picked, np.asarray(self._display_points[picked], dtype=float)

        if self._picker is None:
            return None

        try:
            hit = self._picker.Pick(vx, vy, 0, self._plotter.renderer)
            if not hit:
                self.status_label.setText(tr("pc.pick_miss"))
                return None

            pick_position = np.asarray(self._picker.GetPickPosition(), dtype=float)
            point_id = int(self._picker.GetPointId())
            if (
                self._display_indices is not None
                and 0 <= point_id < len(self._display_indices)
            ):
                display_point = self._picked_display_point(point_id, pick_position)
                return point_id, display_point
            world_point = pick_position
        except Exception as exc:
            logger.debug("点云选点失败: %s", exc)
            self.status_label.setText(tr("pc.pick_miss"))
            return None

        if self._display_indices is None:
            self.status_label.setText(tr("pc.pick_miss"))
            return None
        if world_point is not None and self._display_points is not None:
            distances = np.linalg.norm(self._display_points - world_point.reshape(1, 3), axis=1)
            display_index = int(np.argmin(distances))
            display_point = self._picked_display_point(display_index, world_point)
            return display_index, display_point

        try:
            pick_point = np.asarray(self._plotter.picked_point, dtype=float).reshape(3)
            if self._display_points is not None:
                distances = np.linalg.norm(self._display_points - pick_point.reshape(1, 3), axis=1)
                display_index = int(np.argmin(distances))
                display_point = self._picked_display_point(display_index, pick_point)
                return display_index, display_point
        except Exception:
            pass

        self.status_label.setText(tr("pc.pick_miss"))
        return None

    def _qt_to_vtk_coords(self, event) -> tuple[float, float]:
        x = float(event.position().x())
        y = float(self._plotter.interactor.height() - event.position().y() - 1)
        try:
            render_width, render_height = self._plotter.ren_win.GetSize()
            widget_width = max(1, self._plotter.interactor.width())
            widget_height = max(1, self._plotter.interactor.height())
            x *= float(render_width) / float(widget_width)
            y *= float(render_height) / float(widget_height)
        except Exception:
            pass
        return x, y

    def _picked_display_point(
        self,
        display_index: int,
        pick_position: np.ndarray,
    ) -> np.ndarray:
        if pick_position.shape == (3,) and np.all(np.isfinite(pick_position)):
            return pick_position.astype(float, copy=True)
        return np.asarray(self._display_points[display_index], dtype=float)

    def _pick_nearest_screen_point(self, x: float, y: float) -> Optional[int]:
        if self._display_points is None or len(self._display_points) == 0:
            return None

        try:
            renderer = self._plotter.renderer
            render_width, render_height = self._plotter.ren_win.GetSize()
            best_index = None
            best_dist_sq = float("inf")
            best_depth = float("inf")

            for index, point in enumerate(self._display_points):
                renderer.SetWorldPoint(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    1.0,
                )
                renderer.WorldToDisplay()
                sx, sy, sz = renderer.GetDisplayPoint()
                if (
                    not np.isfinite(sx)
                    or not np.isfinite(sy)
                    or not np.isfinite(sz)
                    or sx < 0.0
                    or sx >= render_width
                    or sy < 0.0
                    or sy >= render_height
                    or sz < 0.0
                    or sz > 1.0
                ):
                    continue

                dist_sq = (float(sx) - x) ** 2 + (float(sy) - y) ** 2
                if dist_sq < best_dist_sq or (
                    math.isclose(dist_sq, best_dist_sq) and float(sz) < best_depth
                ):
                    best_index = index
                    best_dist_sq = dist_sq
                    best_depth = float(sz)

            if best_index is None:
                return None

            point_size = max(1.0, float(self.point_size_spin.value()))
            threshold_px = max(12.0, point_size * 5.0)
            if best_dist_sq > threshold_px * threshold_px:
                logger.debug(
                    "屏幕最近点超过阈值: distance=%.2f px threshold=%.2f px",
                    math.sqrt(best_dist_sq),
                    threshold_px,
                )
                return None
            return int(best_index)
        except Exception as exc:
            logger.debug("屏幕空间点云选点失败: %s", exc)
            return None

    def _select_display_point(self, display_index: int, display_point: np.ndarray):
        if self._point_cloud is None or self._display_indices is None:
            return

        raw_index = int(self._display_indices[display_index])
        camera_point = np.asarray(self._point_cloud.points_xyz_m[raw_index], dtype=float)
        self._selected_display_index = int(display_index)
        self._selected_display_point_m = np.asarray(display_point, dtype=float).reshape(3)
        self._selected_raw_index = raw_index
        self._selected_camera_point_m = camera_point
        self._selected_robot_target_raw_m = camera_point_to_robot_target(camera_point)
        self._target_robot_point_m = np.asarray(
            self._selected_robot_target_raw_m,
            dtype=float,
        ).reshape(3)
        pixels = np.asarray(self._point_cloud.pixels_uv[raw_index], dtype=int)

        if pixels.shape == (2,) and np.all(pixels >= 0):
            self.pixel_value.setText(f"u={int(pixels[0])}, v={int(pixels[1])}")
        else:
            self.pixel_value.setText("--")
        self.camera_point_value.setText(_format_vec_m(camera_point))
        self.move_target_point_cm_value.setText(_format_vec_cm(self._target_robot_point_m))
        self.move_target_point_m_value.setText(_format_vec_m(self._target_robot_point_m))

        self._add_selection_marker(display_index, self._selected_display_point_m)
        self._recompute_target_from_selection()
        self.status_label.setText(tr("pc.point_selected"))

    def _add_selection_marker(
        self,
        display_index: int,
        display_point: Optional[np.ndarray] = None,
    ):
        if not HAS_PYVISTA or self._plotter is None or self._display_points is None:
            return
        if display_point is None:
            display_point = self._display_points[display_index]
        point = np.asarray(display_point, dtype=float)
        if self._selected_actor is not None:
            try:
                self._plotter.remove_actor(self._selected_actor)
            except Exception:
                pass
            self._selected_actor = None

        try:
            marker_points = np.asarray(point, dtype=float).reshape(1, 3)
            marker_size = max(10.0, float(self.point_size_spin.value()) * 6.0)
            self._selected_actor = self._plotter.add_points(
                marker_points,
                color="#ffcc00",
                point_size=marker_size,
                render_points_as_spheres=True,
                name="selected_realsense_point",
            )
            self._plotter.render()
        except Exception:
            pass

    def _recompute_target_from_selection(self):
        self._update_transform_summary()
        if self._selected_robot_target_raw_m is None:
            return

        target_rpy = np.array(
            [
                math.radians(self.rpy_spins[0].value()),
                math.radians(self.rpy_spins[1].value()),
                math.radians(self.rpy_spins[2].value()),
            ],
            dtype=float,
        )
        if self._target_robot_point_m is not None:
            self.move_target_point_cm_value.setText(_format_vec_cm(self._target_robot_point_m))
            self.move_target_point_m_value.setText(_format_vec_m(self._target_robot_point_m))
        corrected_target = apply_tcp_offset_correction(
            self._selected_robot_target_raw_m,
            self._tcp_offset,
            target_rpy,
        )
        self.target_point_cm_value.setText(_format_vec_cm(corrected_target))
        self.target_point_m_value.setText(_format_vec_m(corrected_target))
        self._update_move_button_state()

    def _update_transform_summary(self):
        if not hasattr(self, "transform_summary"):
            return
        matrix_text = np.array2string(
            CAMERA_TO_ROBOT_MATRIX,
            precision=3,
            suppress_small=True,
            separator=", ",
        )
        self.transform_summary.setText(
            tr(
                "pc.transform_summary",
                path=str(CAMERA_TO_ROBOT_CONFIG_PATH),
                matrix=matrix_text,
            )
        )

    def _apply_initial_view(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        try:
            self._plotter.view_xy()
            self._plotter.camera.zoom(1.2)
        except Exception:
            pass

    def _update_pick_mode_state(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        if not hasattr(self, "pick_mode_btn"):
            return
        try:
            if self._picker is not None:
                if self.pick_mode_btn.isChecked() and self._viewer_visible:
                    self._picker.PickFromListOn()
                else:
                    self._picker.PickFromListOff()
        except Exception as exc:
            logger.debug("更新点选模式失败: %s", exc)

    def set_tcp_offset(self, tcp_offset):
        source = [] if tcp_offset is None else list(tcp_offset)
        values = np.asarray(source, dtype=float)
        if values.shape != (6,):
            normalized = np.zeros(6, dtype=float)
            for idx in range(min(6, len(values))):
                normalized[idx] = float(values[idx])
            values = normalized
        self._tcp_offset = values
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def update_current_end_pose(self, end_pose):
        self._current_end_pose = end_pose
        if not self._rpy_initialized:
            self._set_rpy_from_pose(end_pose)

    def set_arm_enabled(self, enabled: bool):
        self._arm_enabled = bool(enabled)
        self._update_move_button_state()

    def _fill_current_rpy(self):
        if self._current_end_pose is None:
            self.status_label.setText(tr("pc.no_pose"))
            return
        self._set_rpy_from_pose(self._current_end_pose)
        self._rpy_initialized = True
        self.status_label.setText(tr("pc.rpy_loaded"))

    def _set_rpy_from_pose(self, pose):
        if pose is None:
            return
        values_deg = [
            math.degrees(float(pose.rx)),
            math.degrees(float(pose.ry)),
            math.degrees(float(pose.rz)),
        ]
        for spin, value in zip(self.rpy_spins, values_deg):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def _mark_rpy_initialized(self):
        self._rpy_initialized = True

    def _on_rpy_changed(self):
        self._mark_rpy_initialized()
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def _on_confirm_move(self):
        if self._target_robot_point_m is None:
            self.status_label.setText(tr("pc.no_target"))
            return
        pose = [
            float(self._target_robot_point_m[0]),
            float(self._target_robot_point_m[1]),
            float(self._target_robot_point_m[2]),
            math.radians(self.rpy_spins[0].value()),
            math.radians(self.rpy_spins[1].value()),
            math.radians(self.rpy_spins[2].value()),
        ]
        duration = float(self.duration_spin.value())
        self.move_l_requested.emit(pose, duration)
        self.status_label.setText(tr("pc.movel_sent"))
        self.log_message.emit(tr("pc.movel_sent"))

    def _clear_selection(self, keep_result: bool = False):
        if not keep_result:
            self._selected_display_index = None
            self._selected_display_point_m = None
            self._selected_raw_index = None
            self._selected_camera_point_m = None
            self._selected_robot_target_raw_m = None
            self._target_robot_point_m = None
            self.pixel_value.setText("--")
            self.camera_point_value.setText("--")
            self.move_target_point_cm_value.setText("--")
            self.move_target_point_m_value.setText("--")
            self.target_point_cm_value.setText("--")
            self.target_point_m_value.setText("--")
        if self._selected_actor is not None and self._plotter is not None:
            try:
                self._plotter.remove_actor(self._selected_actor)
                self._plotter.render()
            except Exception:
                pass
        self._selected_actor = None
        self._update_move_button_state()

    def _update_move_button_state(self):
        can_move = self._arm_enabled and self._target_robot_point_m is not None
        if hasattr(self, "move_btn"):
            self.move_btn.setEnabled(can_move)

    def _set_error(self, message: str):
        self.status_label.setText(message)
        self.error_occurred.emit(message)

    def apply_theme(self):
        if HAS_PYVISTA and self._plotter is not None:
            try:
                sc = SCENE_COLORS[ThemeManager.instance().theme]
                self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])
                self._plotter.render()
            except Exception:
                pass
        elif HAS_PYVISTA and hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            try:
                sc = SCENE_COLORS[ThemeManager.instance().theme]
                self._plotter_placeholder.setStyleSheet(
                    f"color: {sc['subtext']}; font-size: 12px; padding: 12px;"
                )
            except Exception:
                pass

    def retranslate_ui(self):
        self.capture_group.setTitle(tr("pc.capture_group"))
        self.serial_label.setText(tr("pc.serial"))
        self.serial_edit.setPlaceholderText(tr("pc.serial_placeholder"))
        self.width_label.setText(tr("pc.width"))
        self.height_label.setText(tr("pc.height"))
        self.fps_label.setText(tr("pc.fps"))
        self.warmup_label.setText(tr("pc.warmup"))
        self.timeout_label.setText(tr("pc.timeout"))
        self.depth_min_label.setText(tr("pc.depth_min"))
        self.depth_max_label.setText(tr("pc.depth_max"))
        self.stride_label.setText(tr("pc.stride"))
        self.max_points_label.setText(tr("pc.max_points"))
        self.point_size_label.setText(tr("pc.point_size"))
        self.align_check.setText(tr("pc.align_depth"))
        self.color_check.setText(tr("pc.use_color"))
        self.flip_view_check.setText(tr("pc.flip_view"))
        self.pick_mode_btn.setText(tr("pc.pick_mode"))
        self.capture_btn.setText(tr("pc.capture"))

        self.transform_group.setTitle(tr("pc.transform_group"))
        self._update_transform_summary()

        self.result_group.setTitle(tr("pc.result_group"))
        self.pixel_label.setText(tr("pc.pixel"))
        self.camera_point_label.setText(tr("pc.camera_point"))
        self.robot_point_cm_label.setText(tr("pc.move_target_point_cm"))
        self.robot_point_m_label.setText(tr("pc.move_target_point_m"))
        self.target_point_cm_label.setText(tr("pc.target_point_cm"))
        self.target_point_m_label.setText(tr("pc.target_point_m"))
        self.move_group.setTitle(tr("pc.move_group"))
        for label, key in zip(self.rpy_labels, ("pc.rx", "pc.ry", "pc.rz")):
            label.setText(tr(key))
        self.duration_label.setText(tr("pc.duration"))
        self.read_rpy_btn.setText(tr("pc.read_rpy"))
        self.move_btn.setText(tr("pc.confirm_move"))
        if hasattr(self, "no_plot_label"):
            self.no_plot_label.setText(tr("pc.no_pyvista"))
        if hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            self._plotter_placeholder.setText(tr("pc.viewer_loading"))

    def cleanup(self):
        if self._capture_worker is not None and self._capture_worker.isRunning():
            self._capture_worker.requestInterruption()
            self._capture_worker.wait(1500)


__all__ = [
    "RealSensePointPanel",
    "camera_point_to_robot_target",
]
