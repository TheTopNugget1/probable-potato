@echo off
python -m pip install --upgrade pip
python -m pip install pyqt5 opencv-python numpy pyyaml pyserial pupil-apriltags
REM Tkinter is included with standard Python on Windows, but for completeness:
python -m pip install tk
pause