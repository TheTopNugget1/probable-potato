import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, messagebox

import serial

PORT = "COM11"
BAUD = 9600
PING_INTERVAL_S = 1.5
READ_TIMEOUT_S = 0.1

class SerialClient:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.rx_queue = queue.Queue()
        self.running = False
        self.last_pong = 0.0


    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open and (time.time() - self.last_pong) < (PING_INTERVAL_S * 2.5)

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=READ_TIMEOUT_S)
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()
            return True
        except Exception as e:
            self.ser = None
            return False

    def close(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def send_line(self, line):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((line.strip() + "\n").encode("utf-8"))
                return True
            except Exception:
                return False
        return False

    def _reader(self):
        buf = b""
        while self.running and self.ser and self.ser.is_open:
            try:
                chunk = self.ser.read(128)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(errors="replace").strip()
                    if text == "PONG":
                        self.last_pong = time.time()
                    self.rx_queue.put(text)
            except Exception:
                break

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PWM Test - Serial Servo Pulse Sender")
        self.geometry("430x260")
        self.min_us = -90
        self.max_us = 90
        self.mid_us = 0

        self.client = SerialClient(PORT, BAUD)
        self._build_ui()

        self.after(200, self._try_connect)
        self.after(300, self._poll_serial)
        self.after(500, self._ping_loop)

        

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Connection row
        self.status_lbl = ttk.Label(frm, text="Status: Disconnected", foreground="red")
        self.status_lbl.grid(row=0, column=0, sticky="w")
        ttk.Button(frm, text="Reconnect", command=self._try_connect).grid(row=0, column=1, sticky="e")

        # Channel
        ttk.Label(frm, text="Channel (0-15):").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.ch_var = tk.IntVar(value=15)
        self.ch_spin = ttk.Spinbox(frm, from_=0, to=15, width=5, textvariable=self.ch_var)
        self.ch_spin.grid(row=1, column=1, sticky="w", pady=(12, 0))

        # Pulse width
        ttk.Label(frm, text="Pulse width (µs):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.us_var = tk.IntVar(value=self.mid_us)
        self.us_entry = ttk.Entry(frm, textvariable=self.us_var, width=10)
        self.us_entry.grid(row=2, column=1, sticky="w", pady=(8, 0))

        # Slider for convenience
        self.us_scale = ttk.Scale(frm, from_=self.min_us, to=self.max_us, orient="horizontal",
                                  command=self._scale_changed)
        self.us_scale.set(self.mid_us)
        self.us_scale.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # Send button
        self.send_btn = ttk.Button(frm, text="Send Pulse", command=self._send_pulse)
        self.send_btn.grid(row=4, column=0, columnspan=2, pady=(12, 0), sticky="ew")

        # Frequency (optional)
        ttk.Label(frm, text="Frequency (Hz):").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.freq_var = tk.IntVar(value=50)
        ttk.Entry(frm, textvariable=self.freq_var, width=10).grid(row=5, column=1, sticky="w", pady=(10, 0))
        ttk.Button(frm, text="Set Freq", command=self._set_freq).grid(row=6, column=0, columnspan=2, sticky="ew")

        # Debug button
        self.debug_btn = ttk.Button(frm, text="Toggle Debug", command=self._toggle_debug)
        self.debug_btn.grid(row=8, column=0, columnspan=2, pady=(8, 0), sticky="ew")


        # Log
        self.log = tk.Text(frm, height=6, width=50)
        self.log.grid(row=7, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        frm.rowconfigure(7, weight=1)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

    def _scale_changed(self, val):
        try:
            self.us_var.set(int(float(val)))
        except Exception:
            pass

    def _try_connect(self):
        if self.client.connected:
            return
        self.client.close()
        ok = self.client.open()
        if ok:
            self._log("Opened serial on {} @ {}".format(PORT, BAUD))
        else:
            self._log("Failed to open serial on {} @ {}".format(PORT, BAUD))
        self._update_status()

    def _poll_serial(self):
        while not self.client.rx_queue.empty():
            line = self.client.client_rx = self.client.rx_queue.get()
            if line:
                self._log("< " + line)
                if line == "READY":
                    # mark as alive immediately
                    self.client.last_pong = time.time()
        self._update_status()
        self.after(100, self._poll_serial)

    def _ping_loop(self):
        if self.client.ser and self.client.ser.is_open:
            self.client.send_line("PING")
        self.after(int(PING_INTERVAL_S * 1000), self._ping_loop)

    def _send_pulse(self):
        ch = self.ch_var.get()
        angle = self.us_var.get()
        if ch < 0 or ch > 15:
            messagebox.showerror("Error", "Channel must be 0..15")
            return

        if angle < self.min_us or angle > self.max_us:
            if not messagebox.askyesno("Clamp", "Pulse out of range (µs). Clamp and send?"):
                return
            angle = max(self.min_us, min(self.max_us, angle))
            self.us_var.set(angle)
            self.us_scale.set(angle)
        sent = self.client.send_line(f"A {ch} {angle}")
        if sent:
            self._log(f"> A {ch} {angle}")
        else:
            self._log("Send failed; not connected?")
        self._update_status()

    def _toggle_debug(self):
        # Toggle debug state and send to Arduino
        if not hasattr(self, "debug_state"):
            self.debug_state = False
        self.debug_state = not self.debug_state
        state_str = "on" if self.debug_state else "off"
        sent = self.client.send_line(f"BUG {state_str}")
        if sent:
            self._log(f"> BUG {state_str}")
        else:
            self._log("Send failed; not connected?")
        self._update_status()

    def _set_freq(self):
        f = self.freq_var.get()
        if f < 24 or f > 1000:
            messagebox.showerror("Error", "Frequency must be 24..1000 Hz")
            return
        sent = self.client.send_line(f"F {f}")
        if sent:
            self._log(f"> F {f}")
        else:
            self._log("Send failed; not connected?")
        self._update_status()

    def _update_status(self):
        if self.client.connected:
            self.status_lbl.config(text="Status: Connected (True)", foreground="green")
        else:
            self.status_lbl.config(text="Status: Disconnected (False)", foreground="red")

    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def destroy(self):
        try:
            self.client.close()
        finally:
            super().destroy()

if __name__ == "__main__":
    App().mainloop()