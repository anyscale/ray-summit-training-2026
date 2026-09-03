"""Tiny notebook visual helper. Import-safe on workers (no IPython at module level)."""
import base64
import os


def show_gif(path, width=680, caption=None):
    """Display an animated GIF inline. Call at the TOP of a long-running cell
    so it animates while the cell works. No-op (with a note) if the file is
    missing, so a partial checkout never breaks the notebook."""
    from IPython.display import HTML, display

    if not os.path.exists(path):
        print(f"[show_gif] {path} not found - skipping visual")
        return
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    cap = (f'<div style="font-size:12px;color:#8f8f88;font-family:monospace;'
           f'margin:2px 0 8px 2px">{caption}</div>') if caption else ""
    display(HTML(
        f'<img src="data:image/gif;base64,{b64}" width="{width}" '
        f'style="border:1px solid #e7e7e0;border-radius:6px">{cap}'
    ))
