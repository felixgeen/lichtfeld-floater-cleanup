import lichtfeld as lf

from .panels.main_panel import SplatSORPanel

_classes = [SplatSORPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("splat_sor loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("splat_sor unloaded")
