"""
Terminal animations for OpenAccelerator CLI.


"""

import threading
import time

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()


class RunningCatAnimation:
    """Animated running black cat for long operations."""

    def __init__(self):
        self.frames = [
            """
    /\\_/\\
   (  o.o  )
    > ^ <
   /     \\
  (  )-(  )
   ^^   ^^
            """,
            """
    /\\_/\\
   (  ^.^  )
    > - <
   /     \\
  (  ) (  )~
   ^^   ^^
            """,
            """
    /\\_/\\
   (  -.^  )
    > ~ <
   /     \\
  (  )  ( )~
   ^^    ^^
            """,
            """
    /\\_/\\
   (  o.-  )
    > v <
   /     \\
  ( )  (  )~
   ^^    ^^
            """,
        ]

        self.running = False
        self.thread = None

    def start(self, message: str = "Processing..."):
        """Start the cat animation."""
        self.running = True
        self.message = message
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the cat animation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _animate(self):
        """Animation loop."""
        frame_idx = 0
        with Live(console=console, refresh_per_second=4) as live:
            while self.running:
                frame = Text(self.frames[frame_idx], style="bold black")
                message = Text(f"\n{self.message}", style="cyan")
                content = Text.assemble(frame, message)
                live.update(Align.center(content))
                frame_idx = (frame_idx + 1) % len(self.frames)
                time.sleep(0.25)


class ProgressBarWithCat:
    """Enhanced progress bar with cat animation."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.cat = RunningCatAnimation()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cat.stop()

    def update(self, current: int):
        """Update progress and show cat if processing is slow."""
        progress = current / self.total
        if progress < 1.0 and not self.cat.running:
            self.cat.start(f"{self.description} - {progress:.1%} complete")
        elif progress >= 1.0 and self.cat.running:
            self.cat.stop()
            console.print("Complete.")
