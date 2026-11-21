"""Colours and metrics for the curve editor's on-screen interface."""

# surfaces
PANEL = (0.075, 0.08, 0.095, 0.88)
PANEL_EDGE = (1.0, 1.0, 1.0, 0.07)
CHIP = (0.14, 0.15, 0.17, 0.95)
CHIP_HOVER = (0.21, 0.22, 0.25, 0.98)
SCRIM = (0.0, 0.0, 0.0, 0.55)

# accent (warm amber) and states
ACCENT = (1.0, 0.66, 0.24, 1.0)
ACCENT_SOFT = (1.0, 0.66, 0.24, 0.22)
ON = (0.36, 0.78, 0.62, 1.0)
DANGER = (0.93, 0.38, 0.34, 1.0)

# text
TEXT = (0.93, 0.94, 0.96, 1.0)
TEXT_DIM = (0.62, 0.65, 0.70, 1.0)
TEXT_FAINT = (0.45, 0.47, 0.52, 1.0)
TEXT_DARK = (0.08, 0.08, 0.09, 1.0)

# side identity
SIDE_TINT = {'CAGE': (0.45, 0.70, 1.0, 1.0), 'TARGET': (0.98, 0.52, 0.78, 1.0)}
SIDE_LABEL = {'CAGE': "CAGE", 'TARGET': "TARGET"}

# in-scene marks
STROKE = (1.0, 1.0, 1.0, 0.95)
HOVER = (1.0, 1.0, 1.0, 1.0)
BORDER_HINT = (1.0, 0.85, 0.35, 0.9)
ACTIVE_SIDE = (0.25, 0.55, 1.0, 0.95)   # side to draw next

# metrics (logical px, scaled by the UI scale at draw time)
MARGIN = 14.0
RADIUS = 7.0
DOCK_H = 30.0
DOCK_PAD = 4.0
CHIP_GAP = 3.0
GROUP_GAP = 10.0
TEXT_SM = 10.0
TEXT_MD = 11.5
TEXT_LG = 13.0
PICK_PX = 12.0
