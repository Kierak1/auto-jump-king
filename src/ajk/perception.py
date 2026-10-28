"""Frame perception: ground sensing, collision learning, visual map.

Expects these attributes on the controller: cfg, collision, state, logger,
classifier, classifier_solid, classifier_free, surface_scanner, surface_hits,
surface_cells, prev_obs, player_visible, screen_identifier, screen_cell,
screen_changed_at, blocked_takeoffs, last_screen_check, screen_mismatch_count,
ground_sensor, verbose.
"""

from __future__ import annotations

import numpy as np

from .vision import PlayerObs


class PerceptionMixin:
    def _sense_ground(self, frame: np.ndarray, cam_x: float, cam_y: float) -> int:
        if not self.state.grounded:
            return 0
        cells = self.ground_sensor.ground_cells(
            frame,
            cam_x,
            cam_y,
            self.state.x,
            self.state.y,
        )
        for wx, _surface_y in cells:
            self.collision.mark_solid(wx, self.state.y + 3.0)
        return len(cells)

    def _learn_collision(self, obs: PlayerObs, cam_x: float, cam_y: float) -> None:
        left = obs.left + cam_x
        right = obs.right + cam_x
        top = obs.top + cam_y
        bottom = obs.bottom + cam_y
        self.collision.mark_rect_free(left, top, right, bottom)

    def _refresh_visual_map(self, frame: np.ndarray, cam_x: float, cam_y: float) -> None:
        occluded = None
        box = self._occlusion_box(cam_x, cam_y)
        if box is not None:
            occluded = (
                box[0] + cam_x,
                box[1] + cam_y,
                box[2] + cam_x,
                box[3] + cam_y,
            )
        if self.cfg.vision.use_classifier:
            added = self.classifier.observe(
                frame,
                cam_x,
                cam_y,
                self.collision.solid_keys(),
                self.collision.free_keys(),
                occluded=occluded,
            )
            if added and self.logger:
                self.logger.event(
                    "visual_examples",
                    added=added,
                    total=len(self.classifier.examples),
                )
            solid, free = self.classifier.predict(frame, cam_x, cam_y)
            self.classifier_solid = solid
            self.classifier_free = free
        if self.logger:
            self.logger.event(
                "visual_map",
                predicted_solid=len(self.collision.predicted_solid),
                predicted_free=len(self.collision.predicted_free),
            )

    def _occlusion_box(self, cam_x: float, cam_y: float) -> tuple[int, int, int, int] | None:
        if self.prev_obs is not None and self.player_visible:
            obs = self.prev_obs
            return (
                int(obs.left) - 2,
                int(obs.top) - 2,
                int(obs.right) + 2,
                int(obs.bottom) + 2,
            )
        if self.state.seen:
            sx = self.state.screen_x
            sy = self.state.screen_y
            if 0 <= sx <= 480 and 0 <= sy <= 360:
                return (int(sx) - 16, int(sy) - 32, int(sx) + 16, int(sy) + 6)
        return None

    def _refresh_surfaces(self, frame: np.ndarray, cam_x: float, cam_y: float) -> None:
        if not self.cfg.vision.surface_scan:
            return
        cells = self.surface_scanner.scan(frame, self._occlusion_box(cam_x, cam_y))
        for x, y in cells:
            key = (int((x + cam_x) // 8), int((y + cam_y) // 8))
            self.surface_hits[key] = self.surface_hits.get(key, 0) + 1
        confirm = self.cfg.vision.surface_confirm_frames
        self.surface_cells = {k for k, n in self.surface_hits.items() if n >= confirm}
        self.collision.set_predictions(
            self.classifier_solid | self.surface_cells,
            self.classifier_free,
        )

    def _update_screen(self, frame: np.ndarray) -> bool:
        now = self._now()
        if now - self.last_screen_check < self.cfg.vision.screen_check_interval_s:
            return False
        self.last_screen_check = now
        cell = self.screen_identifier.identify_cell(frame)
        if cell is None:
            return False
        if self.screen_cell is None:
            self.screen_cell = cell
            self.screen_changed_at = now
            self.collision.set_cell(cell[0], cell[1])
            self.blocked_takeoffs.clear()
            if self.logger:
                self.logger.event("screen", row=cell[0], col=cell[1])
                self.logger.frame(frame, f"screen_{cell[0]}_{cell[1]}")
            return True
        if cell != self.screen_cell:
            self.screen_mismatch_count += 1
            if self.logger:
                self.logger.event(
                    "screen_mismatch",
                    identified=[cell[0], cell[1]],
                    current=[self.screen_cell[0], self.screen_cell[1]],
                    count=self.screen_mismatch_count,
                )
            if self.verbose:
                print(
                    f"[screen] id {cell} != current {self.screen_cell} "
                    f"(mismatch {self.screen_mismatch_count})"
                )
            if self.screen_mismatch_count >= 3:
                d_row = cell[0] - self.screen_cell[0]
                d_col = cell[1] - self.screen_cell[1]
                if abs(d_row) + abs(d_col) == 1:
                    if self.verbose:
                        print(f"[screen] correcting to {cell} via id")
                    self.screen_mismatch_count = 0
                    self._apply_camera_flip(d_row, d_col)
        else:
            self.screen_mismatch_count = 0
        return False
