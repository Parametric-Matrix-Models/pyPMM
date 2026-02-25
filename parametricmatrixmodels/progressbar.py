import sys
from time import time


class ProgressBar:
    """
    Simple console progress bar
    """

    def __init__(
        self, total: int | float, length: int = 40, extra_info: str = ""
    ) -> None:
        self.total = total
        self.length = length
        self.start(extra_info)

    def start(self, extra_info: str = "") -> None:
        self.last = 0
        self.starttime = time()  # estimate time remaining
        self.longest_str = 0
        self.extra_info = extra_info + (" | " if extra_info else "")

    def update(
        self, raw_progress: int | float, dynamic_info: str = ""
    ) -> None:
        if self.total <= 1e-9:
            return
        progress_frac = raw_progress / self.total
        progress_int = int(progress_frac * self.length)
        elapsed = time() - self.starttime
        est_total_time = elapsed / progress_frac if progress_frac > 1e-9 else 0
        remaining = est_total_time - elapsed
        progress = min(progress_int, self.length)
        if progress >= self.last:
            disp = (
                "\r"
                + self.extra_info
                + dynamic_info
                + (" " if dynamic_info else "")
                + "["
                + "#" * progress
                + " " * (self.length - progress)
                + "] ("
                + str(int(remaining))
                + "s)"
            )

            disp_len = len(disp)
            # pad with spaces to overwrite previous longest line
            diff = self.longest_str - disp_len
            if diff > 0:
                disp += " " * diff
            else:
                self.longest_str = disp_len

            sys.stdout.write(disp)
            sys.stdout.flush()
            self.last = progress

    def end(self, final_info: str = "", newline: bool = False) -> None:
        if self.total <= 1e-9:
            return
        disp = (
            f"\r{self.extra_info}"
            + final_info
            + (" " if final_info else "")
            + f"[{'#' * self.length}] ({int(time() - self.starttime)}s)"
        )

        disp_len = len(disp)
        # pad with spaces to overwrite previous longest line
        diff = self.longest_str - disp_len
        if diff > 0:
            disp += " " * diff
        else:
            self.longest_str = disp_len

        if newline:
            disp += "\n"

        sys.stdout.write(disp)
        sys.stdout.flush()
