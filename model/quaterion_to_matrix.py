"""Backward-compatible import shim for the historical quaternion helper name.

This project originally exposed the frame helpers from a misspelled module
name. The canonical implementation now lives in `model.quaternion_to_matrix`,
but the old import path is kept here so legacy code and tests continue to
import cleanly.
"""

from model.quaternion_to_matrix import *
