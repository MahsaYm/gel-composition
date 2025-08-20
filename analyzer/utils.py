

import hashlib


def color_from_label_name(label: str) -> str:
    """Generate a color based on the label name."""
    # Use a deterministic hash that's consistent across machines
    hash = int(hashlib.md5(label.encode('utf-8')).hexdigest(), 36)
    red = (hash % 256)
    green = ((hash // 256) % 256)
    blue = ((hash // 65536) % 256)
    return f'#{red:02x}{green:02x}{blue:02x}'


def get_marker_from_label_name(label: str) -> str:
    """Generate a marker style based on the label name."""
    # Use a deterministic hash that's consistent across machines
    hash = int(hashlib.md5(label.encode('utf-8')).hexdigest(), 36)
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    return markers[hash % len(markers)]
