"""
Thread-safe / display-safe dashboard output (append-imported by visualizer users).

Why: on WSLg the PyBullet GUI and OpenCV's Qt backend are two independent X11
clients in one process. When the X connection drops, Qt's xcb thread calls
exit() from a non-main thread, which produces
  "QObject::killTimer: Timers cannot be stopped from another thread".
Rules enforced here:
  * cv2 window calls are only ever made from the main thread;
  * any cv2/X11 error permanently downgrades the window to file output instead
    of killing the pipeline;
  * mode='file' never touches X11 at all (headless), but still saves images.
"""
import os
import threading

# MIT-SHM over WSLg is a frequent source of X11 I/O errors with Qt.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

import cv2  # noqa: E402


def display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class DashboardWindow:
    def __init__(self, title="Grasp Detection Dashboard", mode="window",
                 save_dir="outputs/dashboard", size=(900, 580)):
        self.title = title
        self.save_dir = save_dir
        self.mode = mode if mode in ("window", "file", "off") else "window"
        if self.mode == "window" and not display_available():
            print("  [Dashboard] No DISPLAY found → headless file mode.")
            self.mode = "file"
        self._owner = threading.main_thread()
        self._window_open = False
        self._size = size
        if self.mode != "off":
            os.makedirs(self.save_dir, exist_ok=True)

    @property
    def windowed(self):
        return self.mode == "window"

    def _on_main_thread(self):
        return threading.current_thread() is self._owner

    def _downgrade(self, err):
        print(f"  [Dashboard] Display error ({err.__class__.__name__}: {err}) → switching to headless "
              f"file mode; images keep being saved to '{self.save_dir}/'.")
        self.mode = "file"
        self._window_open = False

    def show(self, img_rgb, save_name=None):
        """Displays (main thread only) and always saves the latest frame. Returns saved path."""
        if self.mode == "off":
            return None
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.save_dir, "latest.png")
        cv2.imwrite(path, bgr)
        if save_name:
            path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(path, bgr)
        if self.windowed and self._on_main_thread():
            try:
                if not self._window_open:
                    cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.title, *self._size)
                    self._window_open = True
                cv2.imshow(self.title, bgr)
                cv2.waitKey(1)
            except Exception as e:  # cv2.error, X11 failures
                self._downgrade(e)
        return path

    def poll(self, delay_ms=1):
        """Pumps the GUI event loop; returns key code or -1. Safe in every mode."""
        if not (self.windowed and self._window_open and self._on_main_thread()):
            return -1
        try:
            return cv2.waitKey(delay_ms) & 0xFF
        except Exception as e:
            self._downgrade(e)
            return -1

    def close(self):
        if self._window_open and self._on_main_thread():
            try:
                cv2.destroyWindow(self.title)
                cv2.waitKey(1)
            except Exception:
                pass
        self._window_open = False
