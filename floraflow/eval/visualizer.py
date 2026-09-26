"""4-Layer Multi-Camera VLA Telemetry & Keypoint Visualizer.

Renders synchronized multi-camera rollout frames overlaid with:
1. 2D Spatial Softmax keypoints and 5-step motion comet trails weighted by channel confidence.
2. 3D Forward-Kinematics 16-step future action chunk trajectory ribbons projected into camera frames.
3. Multi-Camera Cross-Attention live modality meters and time-series traces.
4. Physical task telemetry HUD (Spout-to-Pot distance, Can Tilt angle, Gripper state, Water particles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class StepTelemetry:
    """Per-step telemetry snapshot captured during a closed-loop vision rollout."""

    step_idx: int
    cam_images: Dict[str, np.ndarray]
    keypoints: Dict[str, np.ndarray]
    confidences: Dict[str, np.ndarray]
    future_traj_2d: Dict[str, np.ndarray]
    future_traj_valid: Dict[str, np.ndarray]
    attn_weights: np.ndarray
    spout_dist_cm: float
    tilt_deg: float
    gripper_width_cm: float
    particles_in_pot: int
    is_pouring: bool
    success: bool
    latency_ms: float
    can_xy: Tuple[float, float]
    plant_xy: Tuple[float, float]


def project_3d_to_camera_pixels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    points_3d: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world coordinates into 2D pixel coordinates for a MuJoCo camera.

    MuJoCo cameras look along their local -Z axis, with +X pointing right and +Y pointing up.

    Args:
        model: MuJoCo model instance.
        data: MuJoCo data instance containing current camera poses (cam_xpos, cam_xmat).
        cam_name: Name of the camera in the scene XML.
        points_3d: Array of shape (N, 3) in world coordinates (meters).
        width: Target image width in pixels.
        height: Target image height in pixels.

    Returns:
        pixels_2d: Array of shape (N, 2) containing (u, v) pixel coordinates.
        valid_mask: Boolean array of shape (N,) indicating points in front of camera and inside frame.
    """
    cam_id = model.camera(cam_name).id
    cam_pos = data.cam_xpos[cam_id]  # (3,)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)  # columns are camera local axes in world

    # Transform world points into camera local frame: P_cam = R^T @ (P_world - cam_pos)
    rel = points_3d - cam_pos[None, :]
    p_cam = rel @ cam_mat  # (N, 3) where [:, 0]=X_right, [:, 1]=Y_up, [:, 2]=Z_back

    x_c = p_cam[:, 0]
    y_c = p_cam[:, 1]
    z_c = p_cam[:, 2]

    # Points in front of camera have z_c < -1e-3 (depth = -z_c > 0)
    depth = -z_c
    in_front = depth > 1e-3
    safe_depth = np.where(in_front, depth, 1.0)

    fovy_deg = float(model.cam_fovy[cam_id])
    fovy_rad = np.radians(fovy_deg)
    focal_px = (0.5 * height) / np.tan(0.5 * fovy_rad)

    u = (0.5 * width) + focal_px * (x_c / safe_depth)
    v = (0.5 * height) - focal_px * (y_c / safe_depth)

    pixels_2d = np.stack([u, v], axis=-1).astype(np.float32)
    in_bounds = (
        in_front
        & (u >= -8.0)
        & (u <= width + 8.0)
        & (v >= -8.0)
        & (v <= height + 8.0)
    )
    return pixels_2d, in_bounds


def _generate_keypoint_palette(num_keypoints: int) -> List[Tuple[int, int, int]]:
    """Generate distinct high-visibility RGB colors for spatial keypoints."""
    base_colors = [
        (56, 189, 248),   # Sky Blue
        (251, 113, 133),  # Rose
        (52, 211, 153),   # Emerald
        (251, 191, 36),   # Amber
        (167, 139, 250),  # Violet
        (244, 114, 182),  # Pink
        (45, 212, 191),   # Teal
        (250, 204, 21),   # Yellow
        (96, 165, 250),   # Blue
        (248, 113, 113),  # Red
        (163, 230, 53),   # Lime
        (232, 121, 249),  # Fuchsia
        (251, 146, 60),   # Orange
        (129, 140, 248),  # Indigo
        (34, 211, 238),   # Cyan
        (74, 222, 128),   # Green
    ]
    return [base_colors[i % len(base_colors)] for i in range(num_keypoints)]


class VisionRolloutVisualizer:
    """Records and renders 4-layer VLA telemetry dashboards during closed-loop rollouts."""

    CAMERA_COLORS = {
        "third_person_cam": (56, 189, 248),  # Sky Blue
        "overhead_cam": (52, 211, 153),      # Emerald
        "wrist_cam": (251, 113, 133),        # Rose
    }

    CAMERA_LABELS = {
        "third_person_cam": "THIRD-PERSON CAM",
        "overhead_cam": "OVERHEAD CAM",
        "wrist_cam": "WRIST CAM",
    }

    def __init__(
        self,
        evaluator: Any,
        title: str = "FLORAFLOW VLA TELEMETRY",
        subtitle: str = "Run 14 | 3-Cam Spatial Softmax + Flow Matching",
        cam_render_size: int = 256,
        max_steps: int = 200,
        trail_length: int = 5,
    ) -> None:
        self.evaluator = evaluator
        self.env = evaluator.env
        self.model = evaluator.model
        self.cameras: Tuple[str, ...] = tuple(evaluator.cameras)
        self.title = title
        self.subtitle = subtitle
        self.cam_size = cam_render_size
        self.max_steps = max_steps
        self.trail_length = trail_length

        # High-resolution renderer dedicated to visualization so inference renderer stays untouched
        self.viz_renderer = mujoco.Renderer(self.env.model, height=cam_render_size, width=cam_render_size)

        # Isolated MjData struct for 16-step future trajectory forward kinematics
        self.fk_data = mujoco.MjData(self.env.model)
        self._pinch_site_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._arm_qpos_adr = self.env._arm_qpos_adr

        num_kp = self.model.encoders[self.cameras[0]].num_keypoints
        self.kp_palette = _generate_keypoint_palette(num_kp)

        self.history: List[StepTelemetry] = []
        self.rendered_frames: List[Image.Image] = []
        self.seed: int = 0

    def reset(self, seed: int = 0, subtitle: Optional[str] = None) -> None:
        """Clear recorded history before starting a new episode."""
        self.history.clear()
        self.rendered_frames.clear()
        self.seed = seed
        if subtitle is not None:
            self.subtitle = subtitle

    def compute_future_3d_trajectory(self, pred_chunk: np.ndarray) -> np.ndarray:
        """Compute 3D world coordinates of the gripper pinch site across the predicted action chunk.

        Args:
            pred_chunk: Array of shape (H, act_dim) containing predicted target joint angles.

        Returns:
            points_3d: Array of shape (H + 1, 3) starting at current pinch position followed by H future steps.
        """
        horizon = pred_chunk.shape[0]
        points_3d = np.zeros((horizon + 1, 3), dtype=np.float32)

        # Step 0 is the current physical pinch position
        points_3d[0] = self.env.data.site_xpos[self._pinch_site_id].copy().astype(np.float32)

        # Copy current qpos to isolated FK struct
        self.fk_data.qpos[:] = self.env.data.qpos[:]
        for h in range(horizon):
            q_target = pred_chunk[h, :7]
            for j_idx, adr in enumerate(self._arm_qpos_adr):
                self.fk_data.qpos[adr] = float(q_target[j_idx])
            mujoco.mj_kinematics(self.env.model, self.fk_data)
            points_3d[h + 1] = self.fk_data.site_xpos[self._pinch_site_id].copy().astype(np.float32)

        return points_3d

    def make_step_callback(self) -> Callable[[Dict[str, Any]], None]:
        """Return a step callback hook compatible with VisionPolicyEvaluator.run_episode."""

        def _callback(step_data: Dict[str, Any]) -> None:
            step_idx = int(step_data["step_idx"])
            obs = step_data["obs"]
            model_obs = step_data["model_obs"]
            pred_chunk = step_data["pred_chunk"]

            # 1. Extract visual keypoints, confidences, and camera cross-attention weights
            vis_tel = self.model.extract_visual_telemetry(model_obs)

            # 2. Compute 3D future trajectory of the gripper pinch site
            points_3d = self.compute_future_3d_trajectory(pred_chunk)

            # 3. Render high-resolution camera frames and project 3D future trajectory
            cam_images: Dict[str, np.ndarray] = {}
            future_traj_2d: Dict[str, np.ndarray] = {}
            future_traj_valid: Dict[str, np.ndarray] = {}

            for cam in self.cameras:
                self.viz_renderer.update_scene(self.env.data, camera=cam)
                cam_images[cam] = self.viz_renderer.render().copy()
                pts_2d, valid_mask = project_3d_to_camera_pixels(
                    self.env.model,
                    self.env.data,
                    cam,
                    points_3d,
                    width=self.cam_size,
                    height=self.cam_size,
                )
                future_traj_2d[cam] = pts_2d
                future_traj_valid[cam] = valid_mask

            gripper_w_cm = max(0.0, float(np.asarray(obs["gripper_width"]).flat[0]) * 100.0)
            can_xy = (float(obs["can_pos"][0]), float(obs["can_pos"][1]))
            plant_xy = (float(obs["plant_pos"][0]), float(obs["plant_pos"][1]))

            snap = StepTelemetry(
                step_idx=step_idx,
                cam_images=cam_images,
                keypoints=vis_tel["keypoints"],
                confidences=vis_tel["confidences"],
                future_traj_2d=future_traj_2d,
                future_traj_valid=future_traj_valid,
                attn_weights=vis_tel["attn_weights"],
                spout_dist_cm=float(step_data["spout_dist"]) * 100.0,
                tilt_deg=float(step_data["tilt_deg"]),
                gripper_width_cm=gripper_w_cm,
                particles_in_pot=int(step_data["particles_in_pot"]),
                is_pouring=bool(step_data["is_pouring"]),
                success=bool(step_data["success"]),
                latency_ms=float(step_data["latency_ms"]),
                can_xy=can_xy,
                plant_xy=plant_xy,
            )
            self.history.append(snap)
            frame = self.render_dashboard_frame(len(self.history) - 1)
            self.rendered_frames.append(frame)

        return _callback

    @staticmethod
    def _infer_task_phase(snap: StepTelemetry) -> Tuple[str, Tuple[int, int, int]]:
        """Determine human-readable manipulation phase and badge color."""
        if snap.is_pouring or snap.tilt_deg >= 35.0:
            return "POURING WATER", (56, 189, 248)
        if snap.gripper_width_cm < 6.5:
            if snap.spout_dist_cm < 16.0:
                return "ALIGNING SPOUT", (52, 211, 153)
            return "LIFT & TRANSPORT", (251, 191, 36)
        if snap.step_idx < 48:
            return "APPROACH HANDLE", (167, 139, 250)
        return "GRASPING HANDLE", (244, 114, 182)

    def _draw_camera_tile(self, hist_idx: int, cam: str, cam_idx: int) -> Image.Image:
        """Render a single camera viewport overlaid with 3D trajectory ribbon and 2D keypoints."""
        snap = self.history[hist_idx]
        raw_rgb = snap.cam_images[cam]
        tile = Image.fromarray(raw_rgb).convert("RGBA")
        overlay = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = self.cam_size, self.cam_size

        # 1. Draw 3D Future Action Chunk Trajectory Ribbon
        pts_2d = snap.future_traj_2d[cam]
        valid = snap.future_traj_valid[cam]
        n_pts = len(pts_2d)
        for i in range(n_pts - 1):
            if valid[i] and valid[i + 1]:
                frac = i / max(n_pts - 2, 1)
                # Gradient from Cyan (0, 245, 212) to Amber (251, 191, 36)
                r = int((1.0 - frac) * 0 + frac * 251)
                g = int((1.0 - frac) * 245 + frac * 191)
                b = int((1.0 - frac) * 212 + frac * 36)
                p0 = (float(pts_2d[i, 0]), float(pts_2d[i, 1]))
                p1 = (float(pts_2d[i + 1, 0]), float(pts_2d[i + 1, 1]))
                draw.line([p0, p1], fill=(r, g, b, 220), width=3)

        for i in range(0, n_pts, 3):
            if valid[i]:
                px, py = float(pts_2d[i, 0]), float(pts_2d[i, 1])
                rad = 3.5 if i == n_pts - 1 else 2.0
                draw.ellipse(
                    [px - rad, py - rad, px + rad, py + rad],
                    fill=(255, 255, 255, 235),
                    outline=(15, 23, 42, 255),
                )

        # 2. Draw 2D Spatial Softmax Keypoints as HUD Crosshair Reticles (Top-8 Active Channels)
        start_t = max(0, hist_idx - self.trail_length + 1)
        confs = snap.confidences[cam]
        top_k_indices = np.argsort(confs)[-8:]

        for k in top_k_indices:
            conf = float(confs[k])
            if conf < 0.15:
                continue
            color = self.kp_palette[int(k)]
            # Subtle motion trail across recent steps
            trail_pts = []
            for t_idx in range(start_t, hist_idx + 1):
                kp_norm = self.history[t_idx].keypoints[cam][k]
                u = 0.5 * (kp_norm[0] + 1.0) * (w - 1)
                v = 0.5 * (kp_norm[1] + 1.0) * (h - 1)
                trail_pts.append((u, v))

            for t_i in range(len(trail_pts) - 1):
                alpha_trail = int(35 + 85 * ((t_i + 1) / len(trail_pts)))
                draw.line(
                    [trail_pts[t_i], trail_pts[t_i + 1]],
                    fill=(color[0], color[1], color[2], alpha_trail),
                    width=1,
                )

            # Current keypoint HUD crosshair reticle (+) with hollow ring
            u_curr, v_curr = trail_pts[-1]
            norm_conf = float(np.clip((conf - 0.15) / 0.65, 0.35, 1.0))
            arm = 4.5 + 2.0 * norm_conf
            ring_r = 3.0 + 1.5 * norm_conf
            alpha = int(150 + 100 * norm_conf)

            # Dark shadow for contrast against bright table
            draw.line(
                [(u_curr - arm, v_curr), (u_curr + arm, v_curr)],
                fill=(15, 23, 42, alpha),
                width=2,
            )
            draw.line(
                [(u_curr, v_curr - arm), (u_curr, v_curr + arm)],
                fill=(15, 23, 42, alpha),
                width=2,
            )
            # Colored crosshair arms and hollow target ring
            rgba = (color[0], color[1], color[2], alpha)
            draw.line([(u_curr - arm, v_curr), (u_curr + arm, v_curr)], fill=rgba, width=1)
            draw.line([(u_curr, v_curr - arm), (u_curr, v_curr + arm)], fill=rgba, width=1)
            draw.ellipse(
                [u_curr - ring_r, v_curr - ring_r, u_curr + ring_r, v_curr + ring_r],
                fill=None,
                outline=rgba,
                width=1,
            )

        # 3. Draw top translucent camera header bar with live attention weight
        draw.rectangle([0, 0, w, 26], fill=(15, 23, 42, 195))
        cam_label = self.CAMERA_LABELS.get(cam, cam.upper())
        cam_color = self.CAMERA_COLORS.get(cam, (56, 189, 248))
        attn_pct = float(snap.attn_weights[cam_idx]) * 100.0 if cam_idx < len(snap.attn_weights) else 0.0

        draw.rectangle([0, 0, 5, 26], fill=(cam_color[0], cam_color[1], cam_color[2], 255))
        draw.text((10, 6), cam_label, fill=(241, 245, 249, 255))
        attn_text = f"Attn: {attn_pct:4.1f}%"
        draw.text((w - 82, 6), attn_text, fill=(cam_color[0], cam_color[1], cam_color[2], 255))

        # Mini attention bar at bottom of camera header
        bar_w = int((w - 12) * np.clip(attn_pct / 100.0, 0.0, 1.0))
        draw.rectangle([6, 22, w - 6, 24], fill=(51, 65, 85, 180))
        if bar_w > 0:
            draw.rectangle([6, 22, 6 + bar_w, 24], fill=(cam_color[0], cam_color[1], cam_color[2], 255))

        composed = Image.alpha_composite(tile, overlay).convert("RGB")
        return composed

    def render_dashboard_frame(self, hist_idx: int) -> Image.Image:
        """Composite the complete 4-layer dashboard frame for a given history index."""
        snap = self.history[hist_idx]
        n_cams = len(self.cameras)
        pad = 12
        margin = 16
        header_h = 48
        hud_h = 178

        cam_row_w = n_cams * self.cam_size + (n_cams - 1) * pad
        total_w = max(margin * 2 + cam_row_w, 760)
        total_h = header_h + pad + self.cam_size + pad + hud_h

        canvas = Image.new("RGB", (total_w, total_h), (11, 17, 30))
        draw = ImageDraw.Draw(canvas)

        # --- 1. Top Header Banner ---
        draw.rectangle([0, 0, total_w, header_h], fill=(15, 23, 42))
        draw.line([(0, header_h), (total_w, header_h)], fill=(51, 65, 85), width=1)

        draw.text((margin, 9), self.title, fill=(248, 250, 252))
        draw.text((margin, 26), f"{self.subtitle}  |  Seed {self.seed}", fill=(148, 163, 184))

        # Phase pill in center-right
        phase_text, phase_color = self._infer_task_phase(snap)
        phase_box = [total_w - 370, 11, total_w - 220, 37]
        draw.rectangle(phase_box, fill=(30, 41, 59), outline=phase_color, width=1)
        draw.text((phase_box[0] + 12, phase_box[1] + 7), phase_text, fill=phase_color)

        # Step counter
        step_str = f"Step {snap.step_idx + 1:3d}/{self.max_steps}"
        draw.text((total_w - 205, 18), step_str, fill=(226, 232, 240))

        # Status badge (PASS / ACTIVE)
        if snap.success:
            badge_bg = (6, 95, 70)
            badge_border = (16, 185, 129)
            badge_txt = "PASS"
            badge_fg = (167, 243, 208)
        else:
            badge_bg = (30, 41, 59)
            badge_border = (100, 116, 139)
            badge_txt = "ACTIVE" if snap.step_idx < self.max_steps - 1 else "FAIL"
            badge_fg = (226, 232, 240) if snap.step_idx < self.max_steps - 1 else (252, 165, 165)
            if badge_txt == "FAIL":
                badge_bg = (127, 29, 29)
                badge_border = (239, 68, 68)

        badge_box = [total_w - 96, 11, total_w - margin, 37]
        draw.rectangle(badge_box, fill=badge_bg, outline=badge_border, width=1)
        draw.text((badge_box[0] + 18, badge_box[1] + 7), badge_txt, fill=badge_fg)

        # --- 2. Camera Viewport Row ---
        cam_y = header_h + pad
        cam_start_x = (total_w - cam_row_w) // 2
        for idx, cam in enumerate(self.cameras):
            cam_x = cam_start_x + idx * (self.cam_size + pad)
            tile_img = self._draw_camera_tile(hist_idx, cam, idx)
            canvas.paste(tile_img, (cam_x, cam_y))
            draw.rectangle(
                [cam_x - 1, cam_y - 1, cam_x + self.cam_size, cam_y + self.cam_size],
                outline=(51, 65, 85),
                width=1,
            )

        # --- 3. Bottom Telemetry HUD Cards ---
        hud_y = cam_y + self.cam_size + pad
        hud_bottom = total_h - 12
        usable_w = total_w - 2 * margin - 2 * pad
        w1 = int(usable_w * 0.38)
        w2 = int(usable_w * 0.35)

        # Card 1: Spout Alignment & Pour Tilt Time-Series Plot
        c1_x0 = margin
        c1_x1 = c1_x0 + w1
        draw.rectangle([c1_x0, hud_y, c1_x1, hud_bottom], fill=(17, 24, 39), outline=(51, 65, 85))
        draw.text((c1_x0 + 10, hud_y + 7), "SPOUT DIST (cm) & CAN TILT (deg)", fill=(203, 213, 225))
        draw.text(
            (c1_x0 + 10, hud_y + 23),
            f"Dist: {snap.spout_dist_cm:4.1f} cm   Tilt: {snap.tilt_deg:4.1f} deg",
            fill=(56, 189, 248),
        )

        plot1_x0, plot1_y0 = c1_x0 + 12, hud_y + 44
        plot1_x1, plot1_y1 = c1_x1 - 12, hud_bottom - 10
        pw1 = plot1_x1 - plot1_x0
        ph1 = plot1_y1 - plot1_y0
        draw.rectangle([plot1_x0, plot1_y0, plot1_x1, plot1_y1], fill=(11, 17, 30), outline=(30, 41, 59))

        # Target threshold line at 14 cm (out of 60 cm range)
        thresh_y = int(plot1_y1 - (14.0 / 60.0) * ph1)
        for dash_x in range(plot1_x0, plot1_x1, 8):
            draw.line([(dash_x, thresh_y), (min(dash_x + 4, plot1_x1), thresh_y)], fill=(16, 185, 129), width=1)

        if hist_idx >= 1:
            dist_pts = []
            tilt_pts = []
            for t_i in range(hist_idx + 1):
                px = plot1_x0 + (t_i / max(self.max_steps - 1, 1)) * pw1
                d_val = np.clip(self.history[t_i].spout_dist_cm, 0.0, 60.0)
                t_val = np.clip(self.history[t_i].tilt_deg, 0.0, 90.0)
                py_d = plot1_y1 - (d_val / 60.0) * ph1
                py_t = plot1_y1 - (t_val / 90.0) * ph1
                dist_pts.append((px, py_d))
                tilt_pts.append((px, py_t))
            draw.line(dist_pts, fill=(56, 189, 248), width=2)
            draw.line(tilt_pts, fill=(251, 191, 36), width=2)

        # Card 2: Multi-Camera Cross-Attention Telemetry
        c2_x0 = c1_x1 + pad
        c2_x1 = c2_x0 + w2
        draw.rectangle([c2_x0, hud_y, c2_x1, hud_bottom], fill=(17, 24, 39), outline=(51, 65, 85))
        draw.text((c2_x0 + 10, hud_y + 7), "CAMERA CROSS-ATTENTION", fill=(203, 213, 225))

        for idx, cam in enumerate(self.cameras):
            by = hud_y + 26 + idx * 18
            short_name = cam.replace("_cam", "").replace("third_person", "3rd_pers")
            cam_col = self.CAMERA_COLORS.get(cam, (56, 189, 248))
            w_pct = float(snap.attn_weights[idx]) * 100.0 if idx < len(snap.attn_weights) else 0.0
            draw.text((c2_x0 + 10, by), f"{short_name:8s}", fill=(148, 163, 184))
            bx0, bx1 = c2_x0 + 75, c2_x1 - 52
            draw.rectangle([bx0, by + 3, bx1, by + 11], fill=(30, 41, 59))
            fill_w = int((bx1 - bx0) * np.clip(w_pct / 100.0, 0.0, 1.0))
            if fill_w > 0:
                draw.rectangle([bx0, by + 3, bx0 + fill_w, by + 11], fill=cam_col)
            draw.text((c2_x1 - 46, by), f"{w_pct:4.1f}%", fill=cam_col)

        # Attention history trace at bottom of Card 2
        plot2_x0, plot2_y0 = c2_x0 + 10, hud_y + 84
        plot2_x1, plot2_y1 = c2_x1 - 10, hud_bottom - 10
        pw2 = plot2_x1 - plot2_x0
        ph2 = plot2_y1 - plot2_y0
        draw.rectangle([plot2_x0, plot2_y0, plot2_x1, plot2_y1], fill=(11, 17, 30), outline=(30, 41, 59))

        if hist_idx >= 1:
            for idx, cam in enumerate(self.cameras):
                cam_col = self.CAMERA_COLORS.get(cam, (56, 189, 248))
                pts = []
                for t_i in range(hist_idx + 1):
                    px = plot2_x0 + (t_i / max(self.max_steps - 1, 1)) * pw2
                    w_val = float(self.history[t_i].attn_weights[idx])
                    py = plot2_y1 - np.clip(w_val, 0.0, 1.0) * ph2
                    pts.append((px, py))
                draw.line(pts, fill=cam_col, width=2)

        # Card 3: Physical State, Gripper & Water Particles
        c3_x0 = c2_x1 + pad
        c3_x1 = total_w - margin
        draw.rectangle([c3_x0, hud_y, c3_x1, hud_bottom], fill=(17, 24, 39), outline=(51, 65, 85))
        draw.text((c3_x0 + 10, hud_y + 7), "PHYSICAL TELEMETRY", fill=(203, 213, 225))

        grip_state = "GRASPED" if snap.gripper_width_cm < 6.5 else "OPEN"
        grip_col = (52, 211, 153) if grip_state == "GRASPED" else (251, 191, 36)
        draw.text(
            (c3_x0 + 10, hud_y + 28),
            f"Gripper: {snap.gripper_width_cm:3.1f}cm ({grip_state})",
            fill=grip_col,
        )

        # Water Particles in Pot visual droplet meter (8 particles)
        draw.text(
            (c3_x0 + 10, hud_y + 50),
            f"Water in Pot: {snap.particles_in_pot} / 8",
            fill=(226, 232, 240),
        )
        for p_i in range(8):
            dx = c3_x0 + 12 + p_i * 21
            dy = hud_y + 70
            filled = p_i < snap.particles_in_pot
            p_fill = (56, 189, 248) if filled else (30, 41, 59)
            p_out = (125, 211, 252) if filled else (71, 85, 105)
            draw.ellipse([dx, dy, dx + 14, dy + 14], fill=p_fill, outline=p_out, width=1)

        # Legend & Latency
        draw.text(
            (c3_x0 + 10, hud_y + 94),
            f"ODE Latency: {snap.latency_ms:4.1f} ms",
            fill=(148, 163, 184),
        )
        draw.text(
            (c3_x0 + 10, hud_y + 112),
            f"Can XY: ({snap.can_xy[0]:.2f}, {snap.can_xy[1]:.2f})",
            fill=(148, 163, 184),
        )
        draw.text(
            (c3_x0 + 10, hud_y + 130),
            f"Pot XY: ({snap.plant_xy[0]:.2f}, {snap.plant_xy[1]:.2f})",
            fill=(148, 163, 184),
        )

        return canvas

    def save_gif(
        self,
        output_path: str | Path,
        fps: int = 20,
        frame_stride: int = 2,
    ) -> Path:
        """Export recorded dashboard frames as an optimized animated GIF.

        Args:
            output_path: Destination .gif file path.
            fps: Playback frame rate for the full rollout.
            frame_stride: Subsampling stride (e.g. stride=2 saves 100 frames for a 200-step episode).

        Returns:
            out_file: Resolved path to the saved GIF.
        """
        if not self.rendered_frames:
            raise RuntimeError("No frames recorded. Run an episode with make_step_callback() first.")

        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        frames = self.rendered_frames[:: max(1, frame_stride)]
        if self.rendered_frames[-1] is not frames[-1]:
            frames.append(self.rendered_frames[-1])

        duration_ms = int(round((1000.0 / fps) * max(1, frame_stride)))
        frames[0].save(
            out_file,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0,
        )
        return out_file

    def save_summary_strip(
        self,
        output_path: str | Path,
        keyframe_steps: Sequence[int] = (0, 38, 75, 115, 155, 195),
    ) -> Path:
        """Save a high-resolution vertical or 2x3 grid contact sheet of key episode phases."""
        if not self.rendered_frames:
            raise RuntimeError("No frames recorded. Run an episode with make_step_callback() first.")

        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        selected: List[Image.Image] = []
        n_total = len(self.rendered_frames)
        for s in keyframe_steps:
            idx = min(max(0, s), n_total - 1)
            selected.append(self.rendered_frames[idx])

        fw, fh = selected[0].size
        cols = 2
        rows = (len(selected) + cols - 1) // cols
        gap = 12
        grid = Image.new("RGB", (cols * fw + (cols + 1) * gap, rows * fh + (rows + 1) * gap), (7, 11, 20))

        for i, img in enumerate(selected):
            r = i // cols
            c = i % cols
            x = gap + c * (fw + gap)
            y = gap + r * (fh + gap)
            grid.paste(img, (x, y))

        grid.save(out_file)
        return out_file


def save_comparison_gif(
    viz_a: VisionRolloutVisualizer,
    viz_b: VisionRolloutVisualizer,
    output_path: str | Path,
    fps: int = 20,
    frame_stride: int = 2,
) -> Path:
    """Stitch two VisionRolloutVisualizer runs vertically into a synchronized comparison GIF."""
    if not viz_a.rendered_frames or not viz_b.rendered_frames:
        raise RuntimeError("Both visualizers must have recorded frames.")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(viz_a.rendered_frames), len(viz_b.rendered_frames))
    indices = list(range(0, n, max(1, frame_stride)))
    if (n - 1) not in indices:
        indices.append(n - 1)

    combined_frames: List[Image.Image] = []
    w, h = viz_a.rendered_frames[0].size
    gap = 8

    for idx in indices:
        comp = Image.new("RGB", (w, h * 2 + gap), (7, 11, 20))
        comp.paste(viz_a.rendered_frames[idx], (0, 0))
        comp.paste(viz_b.rendered_frames[idx], (0, h + gap))
        combined_frames.append(comp)

    duration_ms = int(round((1000.0 / fps) * max(1, frame_stride)))
    combined_frames[0].save(
        out_file,
        save_all=True,
        append_images=combined_frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
    )
    return out_file
