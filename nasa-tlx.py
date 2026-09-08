#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Raw NASA-TLX with additional custom questions.
#
# Plain Tkinter implementation - no third-party dependencies.
# Behaviour follows TimDomino/nasa-tlx (Raw TLX, no subscale weighting),
# extended with configurable custom Likert items and a free-text field.
#
# Run:  python3 nasa_tlx_tk.py
# ---------------------------------------------------------------------------

import csv
import os
import platform
import tkinter as tk
from tkinter import messagebox, ttk

# ===========================================================================
# CONFIGURATION - normally only this block needs editing
# ===========================================================================

RESULTS_FILE = "nasa-tlx-results.csv"

experiments = ["Fastener tightening"]

conditions = [
    "Camera only",
    "Camera + torque display",
]

# --- The six standard Raw-TLX items (rated 0-100 in steps of 5) ------------
# Numeric values are hidden from the participant, as in the original.

tlx_ids = ["mental", "physical", "temporal", "performance", "effort", "frustration"]

tlx_texts = [
    "Mental Demand - How mentally demanding was the task?",
    "Physical Demand - How physically demanding was the task?",
    "Temporal Demand - How hurried or rushed was the pace of the task?",
    "Performance - How successful were you in accomplishing what you were asked to do?",
    "Effort - How hard did you have to work to accomplish your level of performance?",
    "Frustration - How insecure, discouraged, irritated, stressed and annoyed were you?",
]

tlx_left = ["Very Low", "Very Low", "Very Low", "Failure", "Very Low", "Very Low"]
tlx_right = ["Very High", "Very High", "Very High", "Perfect", "Very High", "Very High"]
# Note: Performance is intentionally the odd one out - higher = better here,
# unlike the other five scales where higher = worse/more demanding. This is
# the opposite of the classic NASA-TLX convention (which puts "Perfect" on
# the low/left end so all six scales combine directly); we flipped it so the
# slider and any plot of tlx_performance read intuitively ("more to the
# right"/"taller bar" = better performance). analyze_results.py accounts for
# this when computing the combined tlx_overall workload score.

# --- Custom questions, appended below the TLX block -----------------------
# Each entry: (column_id, question_text, left_anchor, right_anchor, n_points)
# The numeric value IS shown for these items. Set to [] to disable.

custom_items = [
    ("confidence",
     "How confident were you that the fastener would not break?",
     "Not at all", "Completely", 7),
    ("usefulness",
     "How useful was the feedback for deciding when to stop tightening?",
     "Useless", "Very useful", 7),
    ("naturalness",
     "How immersive did the feedback feel?",
     "Very unnatural", "Very natural", 7),
    
]

# --- Optional free-text field ---------------------------------------------

ASK_FOR_COMMENT = True
COMMENT_PROMPT = "Did you interpreted the haptic feedback as a Torque? (Y or N, X if camera-only)"

# --- Appearance -----------------------------------------------------------

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
FONT_FAMILY = "Helvetica"
FONT_SIZE = 13
ANCHOR_WIDTH = 16          # character width reserved for the scale end labels

BG = "#ffffff"
FG = "#1a1a1a"
FG_MUTED = "#555555"
TROUGH = "#d8d8d8"

# ===========================================================================
# Window scaffolding
# ===========================================================================

root = tk.Tk()
root.title("NASA-TLX")
root.geometry("%dx%d" % (WINDOW_WIDTH, WINDOW_HEIGHT))
root.minsize(700, 480)
root.resizable(True, True)
root.configure(bg=BG)

base_font = (FONT_FAMILY, FONT_SIZE)
bold_font = (FONT_FAMILY, FONT_SIZE, "bold")

# --- Header: experiment, participant id, condition ------------------------

header = tk.Frame(root, bg=BG, padx=16, pady=12)
header.pack(fill="x", side="top")

experiment_var = tk.StringVar(value=experiments[0])
condition_var = tk.StringVar(value=conditions[0])
user_id_var = tk.StringVar(value="1")

tk.Label(header, text="Experiment", bg=BG, fg=FG, font=base_font).grid(
    row=0, column=0, sticky="w", padx=(0, 8))
ttk.Combobox(header, textvariable=experiment_var, values=experiments,
             state="readonly", width=26, font=base_font).grid(
    row=0, column=1, sticky="w", padx=(0, 24))

tk.Label(header, text="User ID", bg=BG, fg=FG, font=base_font).grid(
    row=0, column=2, sticky="w", padx=(0, 8))
tk.Spinbox(header, from_=1, to=200, textvariable=user_id_var, width=6,
           font=base_font).grid(row=0, column=3, sticky="w", padx=(0, 24))

tk.Label(header, text="Condition", bg=BG, fg=FG, font=base_font).grid(
    row=0, column=4, sticky="w", padx=(0, 8))
ttk.Combobox(header, textvariable=condition_var, values=conditions,
             state="readonly", width=34, font=base_font).grid(
    row=0, column=5, sticky="w")

ttk.Separator(root, orient="horizontal").pack(fill="x", side="top")

# --- Footer: submit button, pinned so it is always reachable ---------------

footer = tk.Frame(root, bg=BG, padx=16, pady=12)
footer.pack(fill="x", side="bottom")

# --- Body: scrollable question area ---------------------------------------

container = tk.Frame(root, bg=BG)
container.pack(fill="both", expand=True, side="top")

canvas = tk.Canvas(container, bg=BG, highlightthickness=0, borderwidth=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

body = tk.Frame(canvas, bg=BG)
body_window = canvas.create_window((0, 0), window=body, anchor="nw")

captions = []   # labels whose wraplength tracks the window width


def on_body_configure(_event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))


def on_canvas_configure(event):
    canvas.itemconfigure(body_window, width=event.width)
    wrap = max(300, event.width - 60)
    for label in captions:
        label.configure(wraplength=wrap)


body.bind("<Configure>", on_body_configure)
canvas.bind("<Configure>", on_canvas_configure)


def on_mousewheel(event):
    delta = event.delta
    if platform.system() == "Windows":
        delta = int(delta / 120)
    canvas.yview_scroll(-delta, "units")


canvas.bind_all("<MouseWheel>", on_mousewheel)
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

# ===========================================================================
# Question construction
# ===========================================================================


def add_question(text, left, right, low, high, start, show_value):
    """Add one question block and return its variable."""
    block = tk.Frame(body, bg=BG, padx=24, pady=10)
    block.pack(fill="x")

    caption = tk.Label(block, text=text, bg=BG, fg=FG, font=bold_font,
                       justify="left", anchor="w", wraplength=880)
    caption.pack(fill="x", pady=(0, 6))
    captions.append(caption)

    line = tk.Frame(block, bg=BG)
    line.pack(fill="x")

    tk.Label(line, text=left, width=ANCHOR_WIDTH, anchor="e", bg=BG,
             fg=FG_MUTED, font=base_font).pack(side="left")

    variable = tk.IntVar(value=start)
    scale = tk.Scale(line, from_=low, to=high, orient="horizontal",
                     variable=variable, resolution=1,
                     showvalue=1 if show_value else 0,
                     bg=BG, fg=FG, troughcolor=TROUGH,
                     highlightthickness=0, borderwidth=0, length=420,
                     font=base_font)
    scale.pack(side="left", fill="x", expand=True, padx=10)

    tk.Label(line, text=right, width=ANCHOR_WIDTH, anchor="w", bg=BG,
             fg=FG_MUTED, font=base_font).pack(side="left")

    ttk.Separator(body, orient="horizontal").pack(fill="x", padx=24)
    return variable


tlx_vars = []
for index in range(len(tlx_texts)):
    tlx_vars.append(add_question(
        tlx_texts[index], tlx_left[index], tlx_right[index],
        0, 20, 10, show_value=False))

custom_vars = []
for item_id, text, left, right, points in custom_items:
    midpoint = (1 + points) // 2
    custom_vars.append(add_question(
        text, left, right, 1, points, midpoint, show_value=True))

comment_widget = None
if ASK_FOR_COMMENT:
    comment_block = tk.Frame(body, bg=BG, padx=24, pady=10)
    comment_block.pack(fill="x")
    tk.Label(comment_block, text=COMMENT_PROMPT, bg=BG, fg=FG,
             font=bold_font, anchor="w").pack(fill="x", pady=(0, 6))
    comment_widget = tk.Text(comment_block, height=4, wrap="word",
                             font=base_font, highlightthickness=1,
                             highlightbackground=TROUGH)
    comment_widget.pack(fill="x")

# ===========================================================================
# Submission
# ===========================================================================


def build_header_row():
    columns = ["experiment", "user_id", "condition"]
    columns += ["tlx_" + name for name in tlx_ids]
    columns += [item[0] for item in custom_items]
    if ASK_FOR_COMMENT:
        columns.append("comment")
    return columns


def on_submit():
    raw_id = user_id_var.get().strip()
    if not raw_id.isdigit():
        messagebox.showerror("Invalid User ID",
                             "The User ID must be a whole number.")
        return

    record = [experiment_var.get(), int(raw_id), condition_var.get()]

    # TLX sliders run 0-20, scaled by 5 to give 0-100 in steps of 5
    record += [variable.get() * 5 for variable in tlx_vars]
    record += [variable.get() for variable in custom_vars]

    if ASK_FOR_COMMENT:
        text = comment_widget.get("1.0", "end").strip().replace("\n", " ")
        record.append(text)

    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(build_header_row())
        writer.writerow(record)

    path = os.path.abspath(RESULTS_FILE)
    print("Results appended to %s" % path)
    messagebox.showinfo("Saved", "Responses were recorded.\n\n%s" % path)
    root.destroy()


submit_button = tk.Button(footer, text="Submit", command=on_submit,
                          font=bold_font, highlightthickness=0)
submit_button.pack(side="right")

tk.Label(footer, text="Scroll for further questions.", bg=BG, fg=FG_MUTED,
         font=base_font).pack(side="left")

root.mainloop()