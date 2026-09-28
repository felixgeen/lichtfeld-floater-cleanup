"""Floater Cleanup (SOR) panel for LichtFeld Studio.

Retained RML panel: the DOM is built once from main_panel.rml and only the
bound values update, so sliders are never rebuilt mid-drag.
"""

import threading
import traceback
from pathlib import Path

import numpy as np
import lichtfeld as lf

try:
    from lfs_plugins import ScrubFieldController, ScrubFieldSpec
except ImportError:
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec

from ..sor_core import build_removal_mask

DATA_MODEL = "splat_sor"

SCRUB_FIELDS = {
    "k": ScrubFieldSpec(4, 64, 1, "%d", data_type=int),
    "std_ratio": ScrubFieldSpec(0.5, 5.0, 0.05, "%.2f"),
    "local_ratio": ScrubFieldSpec(1.5, 6.0, 0.05, "%.2f"),
    "passes": ScrubFieldSpec(1, 3, 1, "%d", data_type=int),
    "min_opacity": ScrubFieldSpec(0.0, 0.5, 0.01, "%.2f"),
}

SCOPES = ("selected", "visible")
MODES = ("hide", "select")
METHODS = ("local", "global")


class SplatSORPanel(lf.ui.Panel):
    id = "splat_sor.main_panel"
    label = "Floater Cleanup"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 210
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 100

    def __init__(self):
        # Parameters
        self._values = {"k": 20, "std_ratio": 2.0, "local_ratio": 3.0,
                        "passes": 1, "min_opacity": 0.05}
        self._use_opacity = False
        self._only_selection = False
        self._method = "local"
        self._scope = "visible"
        self._mode = "hide"

        # Job state
        self._thread = None
        self._cancel = False
        self._progress = 0.0
        self._shown_progress = -1.0
        self._pending = None       # results waiting to be applied on the main thread
        self._worker_status = None  # status message set by the worker thread
        self._training = False

        # Display state
        self._status = ""
        self._report = ""
        self._error = ""
        self._last_applied = []    # [(node_name, mask_tensor)] for Restore / Bake
        self._last_removed = 0
        self._ab_latched = False   # "Show original" toggle is on
        self._ab_holding = False   # hold button is pressed
        self._ab_original = False  # original currently on screen
        self._was_busy = False

        self._handle = None
        self._scrub = ScrubFieldController(SCRUB_FIELDS, self._get_scrub, self._set_scrub)

    # ------------------------------------------------------------------ model

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(DATA_MODEL)
        if model is None:
            return

        for prop in SCRUB_FIELDS:
            model.bind(prop, lambda p=prop: self._fmt(p),
                       lambda value, p=prop: self._set_scrub(p, value))

        model.bind("scope", lambda: self._scope, self._set_scope)
        model.bind("mode", lambda: self._mode, self._set_mode)
        model.bind_func("use_opacity", lambda: self._use_opacity)
        model.bind_event("toggle_use_opacity", self._on_toggle_opacity)
        model.bind_func("only_selection", lambda: self._only_selection)
        model.bind_event("toggle_only_selection", self._on_toggle_only_selection)
        model.bind("method", lambda: self._method, self._set_method)
        model.bind_func("method_global", lambda: self._method == "global")

        model.bind_func("locked", lambda: self._busy() or self._training)
        model.bind_func("busy", self._busy)
        model.bind_func("training_active", lambda: self._training and not self._busy())
        model.bind_func("show_run", lambda: not self._busy() and not self._training)
        model.bind_func("can_restore", lambda: bool(self._last_applied) and not self._busy())
        model.bind_func("can_compare", lambda: bool(self._last_applied) and not self._busy())
        model.bind_func("ab_original", lambda: self._ab_original)
        model.bind_func("ab_holding", lambda: self._ab_holding)
        model.bind_func("ab_latched", lambda: self._ab_latched)
        model.bind_func("ab_state", self._ab_state_text)
        model.bind_func("ab_toggle_label",
                        lambda: "Show cleaned" if self._ab_latched else "Show original")
        model.bind_event("ab_press", lambda *_: self._ab_press())
        model.bind_event("ab_release", lambda *_: self._ab_release())
        model.bind_event("toggle_ab", lambda *_: self._ab_toggle())
        model.bind_func("progress_value", lambda: f"{self._progress:.3f}")
        model.bind_func("progress_pct", lambda: f"{self._progress * 100:.0f}%")

        model.bind_func("status", lambda: self._status)
        model.bind_func("has_status", lambda: bool(self._status))
        model.bind_func("report", lambda: self._report)
        model.bind_func("has_report", lambda: bool(self._report))
        model.bind_func("error", lambda: self._error)
        model.bind_func("has_error", lambda: bool(self._error))

        model.bind_event("do_run", lambda *_: self._start())
        model.bind_event("do_cancel", lambda *_: self._request_cancel())
        model.bind_event("do_restore", lambda *_: self._restore())
        model.bind_event("do_bake", lambda *_: self._bake())

        self._handle = model.get_handle()
        self._dirty_all()

    def on_mount(self, doc):
        self._scrub.mount(doc)
        # Catch a release anywhere in the panel, not just over the hold button
        body = doc.get_element_by_id("body")
        if body is not None:
            body.add_event_listener("mouseup", lambda _ev: self._ab_release())

    def on_unmount(self, doc):
        doc.remove_data_model(DATA_MODEL)
        self._handle = None
        self._scrub.unmount()

    def on_update(self, doc):
        del doc
        changed = False

        training = self._training_active()
        if training != self._training:
            self._training = training
            changed = True

        busy = self._busy()
        if busy:
            if abs(self._progress - self._shown_progress) >= 0.005:
                self._shown_progress = self._progress
                self._dirty("progress_value", "progress_pct")
                changed = True
        elif self._was_busy:
            # Worker just finished: apply on the main thread
            self._thread = None
            if self._pending is not None:
                results, self._pending = self._pending, None
                try:
                    self._apply(results)
                except Exception:
                    self._error = traceback.format_exc()
            elif self._worker_status:
                self._status = self._worker_status
            self._worker_status = None
            changed = True

        if busy != self._was_busy:
            self._was_busy = busy
            changed = True

        if changed:
            self._dirty_all()

        changed |= self._scrub.sync_all()
        return changed

    # ------------------------------------------------------------------ values

    def _get_scrub(self, prop):
        return self._values[prop]

    def _set_scrub(self, prop, value):
        if self._busy():
            return  # settings are frozen while a cleanup runs
        spec = SCRUB_FIELDS[prop]
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        v = max(spec.min_value, min(spec.max_value, v))
        self._values[prop] = int(round(v)) if spec.data_type is int else v
        self._dirty(prop)

    def _fmt(self, prop):
        return SCRUB_FIELDS[prop].fmt % self._values[prop]

    def _set_scope(self, value):
        if not self._busy() and str(value) in SCOPES:
            self._scope = str(value)
        self._dirty("scope")

    def _set_mode(self, value):
        if not self._busy() and str(value) in MODES:
            self._mode = str(value)
        self._dirty("mode")

    def _set_method(self, value):
        if not self._busy() and str(value) in METHODS:
            self._method = str(value)
        self._dirty("method", "method_global")

    def _on_toggle_only_selection(self, *_):
        if not self._busy():
            self._only_selection = not self._only_selection
        self._dirty("only_selection")

    def _on_toggle_opacity(self, *_):
        if not self._busy():
            self._use_opacity = not self._use_opacity
        self._dirty("use_opacity")

    def _dirty(self, *names):
        if self._handle:
            for name in names:
                self._handle.dirty(name)

    def _dirty_all(self):
        if self._handle:
            self._handle.dirty_all()

    # ------------------------------------------------------------------ A/B compare

    def _ab_state_text(self):
        if self._ab_original:
            return f"Viewing: ORIGINAL (+{self._last_removed:,} Gaussians)"
        return f"Viewing: CLEANED (-{self._last_removed:,} Gaussians)"

    def _show_original(self, show):
        """Swap between cleaned and original by toggling the soft-delete mask."""
        if show == self._ab_original or not self._last_applied:
            return
        self._set_deleted(self._last_applied, deleted=not show)
        self._ab_original = show
        self._dirty("ab_original", "ab_state")

    def _ab_press(self):
        if self._busy() or not self._last_applied:
            return
        self._ab_holding = True
        self._show_original(True)
        self._dirty("ab_holding")

    def _ab_release(self):
        if not self._ab_holding:
            return
        self._ab_holding = False
        self._show_original(self._ab_latched)
        self._dirty("ab_holding")

    def _ab_toggle(self):
        if self._busy() or not self._last_applied:
            return
        self._ab_latched = not self._ab_latched
        self._show_original(self._ab_latched or self._ab_holding)
        self._dirty("ab_latched", "ab_toggle_label")

    def _ab_reset(self):
        """Put the cleaned result back on screen and clear compare state."""
        self._ab_holding = False
        self._ab_latched = False
        self._show_original(False)

    # ------------------------------------------------------------------ helpers

    def _busy(self):
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def _training_active():
        try:
            return lf.has_trainer() and lf.trainer_state() in ("running", "stopping")
        except Exception:
            return False

    def _set_status(self, text):
        self._status = text
        self._dirty("status", "has_status")

    def _collect_splat_nodes(self, scene, node, out, seen):
        if node is None or node.id in seen:
            return
        seen.add(node.id)
        if node.splat_data() is not None:
            out.append(node)
        for child_id in node.children:
            self._collect_splat_nodes(scene, scene.get_node_by_id(child_id), out, seen)

    def _target_nodes(self, scene):
        out, seen = [], set()
        if self._scope == "selected":
            for name in lf.get_selected_node_names() or []:
                self._collect_splat_nodes(scene, scene.get_node(name), out, seen)
        else:
            for node in scene.get_visible_nodes():
                if node.splat_data() is not None and node.id not in seen:
                    seen.add(node.id)
                    out.append(node)
        return out

    @staticmethod
    def _all_splat_nodes(scene):
        return [n for n in scene.get_nodes() if n.splat_data() is not None]

    # ------------------------------------------------------------------ run

    def _start(self):
        if self._busy() or self._training:
            return
        self._error = ""
        self._report = ""
        scene = lf.get_scene()
        if scene is None:
            self._set_status("Load a trained splat (PLY or checkpoint) first.")
            self._dirty_all()
            return
        nodes = self._target_nodes(scene)
        if not nodes:
            self._set_status("No splat nodes found. Select a splat node in the scene tree."
                             if self._scope == "selected" else "No visible splat nodes found.")
            self._dirty_all()
            return
        single_splat = len(self._all_splat_nodes(scene)) == 1
        if self._only_selection and not single_splat:
            self._set_status("'Only inside selection' needs a scene with exactly one splat node.")
            self._dirty_all()
            return
        if self._mode == "select" and not single_splat:
            self._set_status("'Select outliers only' needs a scene with exactly one splat node. "
                             "Use 'Hide outliers' instead.")
            self._dirty_all()
            return

        if self._last_applied:
            self._ab_reset()
            self._set_deleted(self._last_applied, deleted=False)
            self._last_applied = []

        try:
            region = None
            if self._only_selection:
                if not scene.has_selection():
                    self._set_status("Nothing is selected. Select the area to clean with "
                                     "LichtFeld's selection tools, or untick 'Only inside selection'.")
                    self._dirty_all()
                    return
                region = scene.selection_mask.contiguous().cpu().numpy().reshape(-1).astype(bool)

            jobs = []
            for node in nodes:
                sd = node.splat_data()
                means = sd.means_raw.contiguous().cpu().numpy().reshape(-1, 3)
                if sd.has_deleted_mask():
                    alive = ~sd.deleted.contiguous().cpu().numpy().reshape(-1).astype(bool)
                else:
                    alive = np.ones(len(means), dtype=bool)
                opacity = None
                if self._use_opacity:
                    opacity = sd.get_opacity().contiguous().cpu().numpy().reshape(-1)
                node_region = None
                if region is not None:
                    if len(region) != len(means):
                        raise RuntimeError(
                            f"Selection mask has {len(region):,} entries but the splat has "
                            f"{len(means):,} Gaussians.")
                    node_region = region
                jobs.append((node.name, means, alive, opacity, node_region))
        except Exception:
            self._error = traceback.format_exc()
            self._dirty_all()
            return

        params = {
            "k": int(self._values["k"]),
            "method": self._method,
            "std_ratio": float(self._values["std_ratio"]),
            "local_ratio": float(self._values["local_ratio"]),
            "passes": int(self._values["passes"]),
            "min_opacity": float(self._values["min_opacity"]) if self._use_opacity else None,
        }
        self._cancel = False
        self._progress = 0.0
        self._shown_progress = -1.0
        self._pending = None
        self._worker_status = None
        self._status = f"Analysing {sum(len(j[1]) for j in jobs):,} Gaussians..."
        self._thread = threading.Thread(target=self._worker, args=(jobs, params), daemon=True)
        self._thread.start()
        self._was_busy = True
        self._dirty_all()

    def _request_cancel(self):
        if self._busy():
            self._cancel = True
            self._set_status("Cancelling...")

    def _worker(self, jobs, p):
        # Runs off the main thread: numpy/scipy only, no LichtFeld calls.
        try:
            total = max(1, sum(len(j[1]) for j in jobs))
            done = 0
            results = []
            for name, means, alive, opacity, region in jobs:
                n = len(means)

                def prog(f, base=done, n=n):
                    self._progress = (base + f * n) / total

                r = build_removal_mask(
                    means, alive, opacity, p["k"], p["method"], p["std_ratio"],
                    p["local_ratio"], p["passes"], min_opacity=p["min_opacity"],
                    region=region, progress=prog,
                    cancel=lambda: self._cancel,
                )
                if r is None:
                    self._worker_status = "Cancelled. Nothing was changed."
                    return
                r["name"] = name
                results.append(r)
                done += n
            self._pending = results
        except Exception:
            self._worker_status = "Cleanup failed - see the error below."
            self._error = traceback.format_exc()
        finally:
            self._progress = 1.0

    # ------------------------------------------------------------------ apply

    def _apply(self, results):
        scene = lf.get_scene()
        if scene is None:
            self._status = "The scene was closed before the result could be applied."
            return

        applied, lines, removed_total = [], [], 0
        for r in results:
            node = scene.get_node(r["name"])
            sd = node.splat_data() if node is not None else None
            if sd is None or sd.num_points != r["n_total"]:
                lines.append(f"{r['name']}: skipped (node changed while analysing)")
                continue

            n_remove = int(r["mask"].sum())
            removed_total += n_remove
            pct = 100.0 * n_remove / max(1, r["n_alive"])
            lines.append(f"{r['name']}: {n_remove:,} of {r['n_alive']:,} ({pct:.2f}%)")
            lines.append(f"   outliers {r['n_sor']:,} | low opacity {r['n_opacity']:,}"
                         f" | invalid {r['n_nonfinite']:,}")
            for s in r["stats"]:
                if s["method"] == "local":
                    lines.append(f"   pass {s['pass']}: median ratio {s['median_score']:.2f}, "
                                 f"99th pct {s['p99_score']:.2f}, threshold {s['threshold']:.2f}, "
                                 f"removed {s['removed']:,}")
                else:
                    lines.append(f"   pass {s['pass']}: mean dist {s['mean_dist']:.4g}, "
                                 f"threshold {s['threshold']:.4g}, removed {s['removed']:,}")

            if n_remove == 0:
                continue
            mask_t = lf.Tensor.from_numpy(r["mask"]).cuda()
            if self._mode == "hide":
                sd.soft_delete(mask_t)
                applied.append((r["name"], mask_t))
            else:
                scene.set_selection_mask(mask_t)

        scene.notify_changed()
        self._report = "\n".join(lines)

        if self._mode == "hide":
            self._last_applied = applied
            self._last_removed = removed_total
            self._ab_holding = self._ab_latched = self._ab_original = False
            self._status = f"Hid {removed_total:,} Gaussians. Use Restore to bring them back."
            self._push_undo(applied)
        else:
            self._status = (f"Selected {removed_total:,} outliers. Inspect them, then delete "
                            "with LichtFeld's own delete tool.")
            if self._only_selection:
                self._status += (" Your selection now holds the outliers, so reselect the area "
                                 "before running again.")

    def _push_undo(self, applied):
        if not applied:
            return
        def undo():
            self._ab_holding = self._ab_latched = self._ab_original = False
            self._set_deleted(applied, deleted=False)
            if self._last_applied is applied:
                self._last_applied = []
            self._dirty_all()

        def redo():
            self._ab_holding = self._ab_latched = self._ab_original = False
            self._set_deleted(applied, deleted=True)
            self._last_applied = applied
            self._dirty_all()

        try:
            lf.undo.push("SOR floater cleanup", undo, redo)
        except Exception:
            pass  # Restore button still works if the undo stack rejects the step

    def _set_deleted(self, applied, deleted):
        scene = lf.get_scene()
        if scene is None:
            return
        for name, mask_t in applied:
            node = scene.get_node(name)
            sd = node.splat_data() if node is not None else None
            if sd is None:
                continue
            if deleted:
                sd.soft_delete(mask_t)
            else:
                sd.undelete(mask_t)
        scene.notify_changed()

    def _restore(self):
        if self._busy():
            return
        self._ab_reset()
        self._set_deleted(self._last_applied, deleted=False)
        self._last_applied = []
        self._status = "Restored the Gaussians hidden by the last cleanup."
        self._dirty_all()

    def _bake(self):
        if self._busy():
            return
        scene = lf.get_scene()
        if scene is None:
            return
        self._ab_reset()  # bake the cleaned state, never the original
        removed = 0
        try:
            for name, _ in self._last_applied:
                node = scene.get_node(name)
                sd = node.splat_data() if node is not None else None
                if sd is not None:
                    removed += int(sd.apply_deleted())
            scene.notify_changed()
            self._status = f"Permanently removed {removed:,} Gaussians. This cannot be restored."
        except Exception:
            self._error = traceback.format_exc()
        self._last_applied = []
        self._dirty_all()
