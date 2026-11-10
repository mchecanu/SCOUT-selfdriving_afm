"""
Asylum Research / Igor Pro control, over COM (win32com, "IgorPro.Application") -- the
steps that were done by hand around every run in Run_SelfDriving_Scan.ipynb:

    afm = asylum.AsylumControl(igor, backend, cfg, executor)
    afm.setup()            # before any scan: centre, AO0/AO1 -> 0, crosspoint + X/Y loops, engage
    with afm.run("grid_refine"):     # engaged for the run; withdrawn + back at (0, 0) after it,
        modes.run(session)           # also if it fails or is interrupted
    afm.finish()           # withdraw + smoothly back to (0, 0) (after killing a run cell)

setup(), step by step (each printed, with the Igor commands it sends):
  0. withdraw (DoScanFunc("StopEngageButton")) -- nothing moves laterally while engaged
  1. tip to Igor's designated spot (spot 0, the centre of the image): Go There,
     DoForceFunc("GoForce_1") (as in afm_laser_sweep_automation), then
     cfg.asylum.center_transit_s to get there. Skipped once the X/Y loops follow the
     FPGA (step 3) in this session -- Igor's own move would act on those loops
  2. the FPGA's AO0/AO1 (PyScanner get_AO0_V / get_AO1_V) at 0 -- if not, moved there
     smoothly (PyScanner's line-scan move, at the scan tip speed)
  3. crosspoint: InA <- BNCIn0, InB <- BNCIn1 (AO0/AO1 come in there), locked, written;
     then the X/Y loops follow inputs A/B -- the two ir_SetPISLoop lines of
     "Cypher XY Control Igor Command.txt" (Cypher by default, cfg.asylum.controller)
  3b/3c. (cfg.position_sensors) XSensor/YSensor copied to BNCOut0/1, calibrated
  3d. (cfg.z_signal.route) the chosen Z signal (ZSensor by default) copied to BNCOut2
     through a Cypher loop (OutFastA preferred) -- the readback AI stays cabled to that BNC,
     and is checked to read the copy
  4. engage (SimpleEngageMe(""))

Without Igor (igor=None: simulation, or a dry run) every command is printed, not sent.
The command names come from the scripts on this instrument (ralph_tools, the laser-sweep
and 3DPFM notebooks, StroboSpectra: CypherXPTLock10Box_0 = BNCOut0, 12 = BNCOut2 -> the
panel's order below); the InA/InB popups follow the same pattern -- CONFIRM ON HARDWARE.
"""

from __future__ import annotations
import contextlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig

# crosspoint panel rows, in the panel's order (= the lock box numbers), per controller
CYPHER_XPT = ["InA", "InB", "InC", "InFastA", "InFastB", "ContInX", "ContInY", "ContInZ", "ZZHV",
              "PFMHV", "BNCOut0", "BNCOut1", "BNCOut2", "BNCOut3", "BNCOut4", "ContDefl"]
ARC_XPT = ["InA", "InB", "InFast", "InAOffset", "InBOffset", "InFastOffset", "OutXMod", "OutYMod",
           "OutZMod", "FilterIn", "BNCOut0", "BNCOut1", "BNCOut2", "PogoOut", "Chip", "Shake"]
PANEL = "CrosspointPanel"
# all 32 Cypher (backpack) crosspoint rows, in order (row i = lock bit 2**i)
CYPHER_XPT_ALL = CYPHER_XPT + ["ContPogoIn0", "ContPogoIn1", "bdDrive", "ExpOut0", "ExpOut1", "ExpOut2", "Math0",
                               "Math1", "Math2", "Math3", "Math4", "Headphone", "Sample", "HolderOut0",
                               "HolderOut1", "HolderOut2"]
SENSOR_DACS = ["OutC", "OutD", "OutA", "OutB", "OutFastA", "OutFastB"]   # Cypher DACs a sensor copy may use
ARC_DACS = ["OutA", "OutB", "OutC"]      # the ARC's user DACs (PFM phase copy) -- CONFIRM ON HARDWARE


def _side_tables(side: str):
    """(crosspoint rows, DACs, popup prefix, lock-box prefix) of the Cypher or the ARC."""
    if side.casefold() == "arc":
        return ARC_XPT, ARC_DACS, "", "XPTLock"
    return CYPHER_XPT, SENSOR_DACS, "Cypher", "CypherXPTLock"


def _side_name(side: str) -> str:
    return "ARC" if side.casefold() == "arc" else "Cypher"
PYDATA = "root:PyData"


def connect_igor(dispatch_name: str = "IgorPro.Application", warn_after_s: float = 15.0):
    """The running Igor Pro (Asylum Research software), over COM. COM connects to the
    FIRST Igor that registered itself -- with two Igor.exe running (e.g. an old one whose
    window is gone) that can be the wrong one, and the call then never returns: after
    warn_after_s a message says so (the call itself can't be cancelled)."""
    import threading
    import win32com.client
    done = threading.Event()

    def _warn():
        if not done.wait(warn_after_s):
            print(f"Still connecting to Igor after {warn_after_s:g} s. If this doesn't finish: is "
                  f"more than one Igor.exe running (Task Manager -> Details)? COM attaches to the "
                  f"first one started, which may be an old instance without a window. Close the "
                  f"extra one, restart the kernel and run this cell again.", flush=True)

    threading.Thread(target=_warn, daemon=True).start()
    try:
        igor = win32com.client.Dispatch(dispatch_name)
    finally:
        done.set()
    print("Connected to Igor Pro")
    return igor


def xy_loop_commands(cfg: ExperimentConfig) -> List[str]:
    """The two Igor lines that make the X/Y piezo loops follow inputs A/B."""
    a = cfg.asylum
    src = "Cypher" if a.controller == "cypher" else "Arc"
    P, I, S = a.xy_loop_gains
    lo, hi = a.xy_loop_output_range
    return [f'ir_SetPISLoop({k},"always,never","{src}.Input.{inp}",{a.xy_loop_setpoint:g},{P:g},{I:g},{S:g},'
            f'"Output.{out}",{lo:g},{hi:g})' for k, inp, out in ((0, "A", "X"), (1, "B", "Y"))]


class AsylumControl:
    def __init__(self, igor, backend, cfg: ExperimentConfig, executor=None, verbose: bool = True):
        self.igor = igor                 # None: print the commands only
        self.backend = backend
        self.cfg = cfg
        self.executor = executor
        self.verbose = verbose
        self.xy_from_fpga = False        # step 3 done in this session
        self.setup_done = False
        self.engaged = False
        self.sensor_cal = None
        self.z_info = None               # route_z(): what the readback AI carries

    # ------------------------------------------------------------------
    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

    def cmd(self, command: str) -> None:
        """Send one Igor command (printed first)."""
        self._say(f"    Igor> {command}" + ("" if self.igor is not None else "   [no Igor: not sent]"))
        if self.igor is not None:
            self.igor.Execute(command)

    def ex(self, control: str, panel: str, val=0, string: str = "") -> None:
        """ARExecuteControl -- press / set a control on an Asylum panel."""
        self.cmd(f'ARExecuteControl("{control}", "{panel}", {val}, "{string}")')

    def master_variables(self) -> Optional[Dict[str, float]]:
        """Igor's MasterVariablesWave as {label: value} (None without Igor)."""
        if self.igor is None:
            return None
        w = self.igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
        return {w.DimensionLabel(0, i, 0): w.GetNumericWavePointValue(i) for i in range(w.GetDimensions()[1])}

    # ------------------------------------------------------------------
    # the pieces
    # ------------------------------------------------------------------
    def engage(self) -> None:
        self._say("  Engaging (SimpleEngageMe) ...")
        self.cmd('SimpleEngageMe("")')
        time.sleep(self.cfg.asylum.engage_wait_s if self.igor is not None else 0)
        self.engaged = True
        self._say(f"  Engaged (waited {self.cfg.asylum.engage_wait_s:g} s)")

    def withdraw(self) -> None:
        self._say("  Withdrawing ...")
        self.cmd('DoScanFunc("StopEngageButton")')
        time.sleep(self.cfg.asylum.withdraw_wait_s if self.igor is not None else 0)
        self.engaged = False
        self._say("  Withdrawn")

    def center(self) -> None:
        """Tip to Igor's designated spot (spot 0, the centre of the image): the force
        panel's Go There -- DoForceFunc("GoForce_1"), as in afm_laser_sweep_automation."""
        a = self.cfg.asylum
        if self.xy_from_fpga:
            self._say("  Centre: skipped -- the X/Y loops already follow the FPGA (AO0/AO1) in this session")
            return
        self._say("  Moving the tip to the designated spot (Go There, GoForce) ...")
        self.cmd('DoForceFunc("GoForce_1")')
        time.sleep(a.center_transit_s if self.igor is not None else 0)
        self._say(f"  At the spot (waited {a.center_transit_s:g} s)")

    def ao_xy(self) -> Optional[Tuple[float, float]]:
        read = getattr(self.backend, "read_ao_xy", None)
        return read() if callable(read) else None

    def zero_ao(self, label: str = "FPGA AO0/AO1") -> None:
        """AO0/AO1 to (0, 0), smoothly, if they aren't there already."""
        v = self.cfg.velocity.tip_velocity_V_s
        now = self.ao_xy()
        if now is None:
            # can't read them: move anyway, slowly enough for the farthest they can be
            T = max(np.sqrt(2) * self.cfg.safety.max_amplitude_V / v, 0.2)
            self._say(f"  {label}: can't read them -- moving to (0, 0) anyway over {T:.2f} s")
            self.backend.move_to(0.0, 0.0, transition_time_s=T)
        else:
            d = float(np.hypot(*now))
            if hasattr(self.backend, "last_out"):
                self.backend.last_out = np.array(now, dtype=float)   # the move starts where they really are
            if d <= self.cfg.asylum.ao_zero_tol_V:
                self._say(f"  {label} at ({now[0]:+.4f}, {now[1]:+.4f}) V -- already at 0")
                if getattr(self.backend, "last_out", 0) is None:
                    # (a zero-length move) so the backend knows where its outputs are: its
                    # next moves are smooth waveforms from here
                    self.backend.move_to(0.0, 0.0, transition_time_s=0.2)
            else:
                T = max(d / v, 0.2)
                self._say(f"  {label} at ({now[0]:+.4f}, {now[1]:+.4f}) V -- moving smoothly to (0, 0) "
                          f"over {T:.2f} s ({v:g} V/s)")
                self.backend.move_to(0.0, 0.0, transition_time_s=T)
            after = self.ao_xy()
            if after is not None:
                if d > self.cfg.asylum.ao_zero_tol_V and float(np.hypot(after[0] - now[0], after[1] - now[1])) < 1e-6:
                    self._say(f"  {label}: the readback still says ({after[0]:+.4f}, {after[1]:+.4f}) V -- that "
                              f"indicator doesn't follow the move (the move to (0, 0) was sent)")
                else:
                    self._say(f"  {label} now ({after[0]:+.4f}, {after[1]:+.4f}) V")
        if self.executor is not None:
            self.executor.last_pos = np.zeros(2)   # history untouched (the save cell needs it)

    def set_crosspoint(self) -> None:
        a = self.cfg.asylum
        rows, prefix, lock_var = ((CYPHER_XPT, "Cypher", "CypherXPTLock") if a.controller == "cypher"
                                  else (ARC_XPT, "", "XPTLock"))
        self._say("  Crosspoint: " + ", ".join(f"{k} <- {s}" for k, s in a.crosspoint.items())
                  + (" (locked)" if a.lock_crosspoint else ""))
        for name, signal in a.crosspoint.items():
            if name not in rows:
                raise ValueError(f"crosspoint row {name!r} not known for {a.controller} (known: {rows})")
            self.ex(f"{prefix}{name}Popup", PANEL, 0, signal)
            if a.lock_crosspoint:
                self.ex(f"{prefix}XPTLock{rows.index(name)}Box_0", PANEL, 1, "")
        self.ex("WriteXPT", PANEL, 0, "")
        g = self.master_variables()
        if g is not None and a.lock_crosspoint and lock_var in g:
            bits = int(g[lock_var])
            missing = [n for n in a.crosspoint if not bits >> rows.index(n) & 1]
            self._say(f"  {lock_var} = {bits} -> " + ("all locked" if not missing else f"NOT locked: {missing}"))

    def route_xy_to_fpga(self) -> None:
        self._say("  X/Y piezo loops follow inputs A/B (the FPGA's AO0/AO1):")
        for c in xy_loop_commands(self.cfg):
            self.cmd(f"print {c}")       # Igor prints the return value in its history
        self.xy_from_fpga = True

    # ------------------------------------------------------------------
    # reading values back from Igor (COM Execute returns nothing: through globals in
    # root:PyData) -- None without Igor
    # ------------------------------------------------------------------
    def _num(self, expr: str):
        if self.igor is None:
            return None
        self.igor.Execute(f"NewDataFolder/O {PYDATA}")
        self.igor.Execute(f"Variable/G {PYDATA}:pyNum = {expr}")
        v = self.igor.DataFolder(PYDATA).Variable("pyNum").GetNumericValue()
        return float(v[0] if isinstance(v, (tuple, list)) else v)

    def _str(self, expr: str):
        if self.igor is None:
            return None
        self.igor.Execute(f"NewDataFolder/O {PYDATA}")
        self.igor.Execute(f"String/G {PYDATA}:pyStr = {expr}")
        return str(self.igor.DataFolder(PYDATA).Variable("pyStr").GetStringValue(0))

    def read_value(self, address: str):
        return self._num(f'ir_ReadValue("{address}")')

    def read_route(self, row: str, side: str = "Cypher"):
        """Where crosspoint row `row` (of the Cypher, or the ARC) is routed now (None without Igor)."""
        return self._str(f'ir_ReadString("{_side_name(side)}.crosspoint.{row}")')

    def read_loop(self, k: int, side: str = "Cypher") -> Optional[Dict[str, str]]:
        """td_ReadGroup of <side>.PIDSLoop.k -> {label: value} (None without Igor)."""
        if self.igor is None:
            return None
        self.igor.Execute(f"NewDataFolder/O {PYDATA}")
        self.igor.Execute(f"Make/T/O/N=0 {PYDATA}:pyGroup")
        err = self._num(f'td_ReadGroup("{side}.PIDSLoop.{k}", {PYDATA}:pyGroup)')
        if err:
            raise RuntimeError(f"td_ReadGroup({side}.PIDSLoop.{k}) failed: error {err:g}")
        w = self.igor.DataFolder(PYDATA).Wave("pyGroup")
        return {w.DimensionLabel(0, i, 0): w.GetTextWavePointValue(i, 0) for i in range(int(w.GetDimensions()[1]))}

    def write_loop(self, k: int, group: Dict[str, object], side: str = "Cypher") -> None:
        sn = _side_name(side)
        self._say(f"    Igor> ir_WriteGroup(\"{sn}.PIDSLoop.{k}\", ...)  "
                  + ", ".join(f"{a}={b}" for a, b in group.items() if a.casefold() in _LOOP_KEYS)
                  + ("" if self.igor is not None else "   [no Igor: not sent]"))
        if self.igor is None:
            return
        self.igor.Execute(f"Make/T/O/N={len(group)} {PYDATA}:pyGroup")
        w = self.igor.DataFolder(PYDATA).Wave("pyGroup")
        for i, (a, b) in enumerate(group.items()):
            self.igor.Execute(f'SetDimLabel 0,{i},$"{a}",{PYDATA}:pyGroup')
            w.SetTextWavePointValue(i, 0, str(b))
        codes = self._str(f'ir_WriteGroup("{sn}.PIDSLoop.{k}", {PYDATA}:pyGroup)') or ""
        err = next((int(float(c)) for c in codes.split(",") if c.strip() and float(c) != 0), 0)
        if err:
            raise RuntimeError(f"ir_WriteGroup({sn}.PIDSLoop.{k}) failed: error {err}")

    # ------------------------------------------------------------------
    # is the routing still there? (an Igor scan / mode change resets it: seen 2026-10-08 --
    # loops 0/1 back on LVDT.X/Y, InA/InB back on ACDefl/Defl, unlocked)
    # ------------------------------------------------------------------
    def xy_routing_ok(self) -> Optional[bool]:
        """Crosspoint InA/InB on the FPGA's BNCs and loops 0/1 following inputs A/B? None
        without Igor (can't tell)."""
        if self.igor is None:
            return None
        a = self.cfg.asylum
        for row, sig in a.crosspoint.items():
            if str(self.read_route(row)).casefold() != sig.casefold():
                return False
        # ir_SetPISLoop(0/1, ...) sets the ARC's PIDS loops 0/1 (seen 2026-10-09: ARC loop 0 =
        # Cypher.Input.A -> Output.X); the Cypher's loops 0/1 are Igor's own LVDT.X/Y -> Output.A/B
        for k, inp, out in ((0, "Input.A", "Output.X"), (1, "Input.B", "Output.Y")):
            g = self.read_loop(k, side="ARC")
            if inp.casefold() not in str(_g(g, "InputChannel")).casefold() \
                    or out.casefold() not in str(_g(g, "OutputChannel")).casefold():
                return False
        return True

    def sensors_ok(self) -> Optional[bool]:
        """XSensor/YSensor still copied to their BNCs (loop running, BNC on that DAC)?"""
        if self.igor is None:
            return None
        ps = self.cfg.position_sensors
        for sensor, k, bnc in (("XSensor", ps.x_loop, ps.x_bnc), ("YSensor", ps.y_loop, ps.y_bnc)):
            g = self.read_loop(k)
            out = str(_g(g, "OutputChannel")).split(".")[-1]
            if not _same_signal(_g(g, "InputChannel"), sensor) \
                    or not _active(g) \
                    or str(self.read_route(bnc)).casefold() != f"out{out}".casefold() \
                    or not _p_ok(g, ps.copy_gain)                     or not _setpoint_zero(g):          # Igor leaves the loop's setpoint at the sensor's value
                return False
            if not self._copy_matches(sensor, out, ps.copy_gain):
                return False
        return True

    def _copy_matches(self, sensor: str, dac_letter: str, gain: float, tol_V: float = 0.05,
                      verbose: bool = False, side: str = "Cypher") -> bool:
        """Is the DAC really at -gain x the sensor (the copy absolute, not offset)?"""
        try:
            sv = self.read_value(sensor)
            dv = self.read_value(f"{_side_name(side)}.Output.{dac_letter.upper()}")
        except Exception:     # noqa: BLE001 -- can't read: don't block
            return True
        if sv is None or dv is None or not (np.isfinite(sv) and np.isfinite(dv)):
            return True
        ok = abs(dv - (-gain) * sv) <= tol_V
        if verbose or not ok:
            print(f"    {sensor} {sv:+.4f} V -> Output.{dac_letter.upper()} {dv:+.4f} V "
                  f"(expected {(-gain) * sv:+.4f} V){'' if ok else '  <-- NOT a copy'}")
        return ok

    def ensure_routing(self) -> None:
        """Before a run (tip withdrawn): X/Y loops + crosspoint, and the sensor routing, checked
        and redone if Igor has reset them."""
        ok = self.xy_routing_ok()
        if ok is False:
            now = self.ao_xy()
            if now is not None and float(np.hypot(*now)) > self.cfg.asylum.ao_zero_tol_V:
                raise RuntimeError(f"Igor has reset the X/Y routing and AO0/AO1 are at ({now[0]:+.4f}, {now[1]:+.4f}) V, "
                                   f"not 0: handing the loops back to them would jump -- run afm.zero_ao() first")
            print("  Igor has reset the X/Y routing (an Igor scan / mode change does) -- setting it again:")
            self.set_crosspoint()
            self.route_xy_to_fpga()
        elif ok is True:
            self._say("  X/Y routing to the FPGA: still in place")
        if self.cfg.position_sensors.enabled:
            # a calibration measured with another copy gain (divide_output changed since) is wrong now
            for owner in (self, self.executor, self.backend):
                cal = getattr(owner, "sensor_cal", None)
                if cal is not None and abs(cal.report.get("copy_gain", -1.0) - self.cfg.position_sensors.copy_gain) > 1e-9:
                    print("  WARNING: the sensor calibration was measured with a different output divider -- "
                          "dropped; rerun the setup cell to calibrate again (until then: commanded positions)")
                    owner.sensor_cal = None
            ok = self.sensors_ok()
            if ok is False:
                print("  Igor has reset the sensor routing -- setting it again:")
                self.route_sensors()
            elif ok is True:
                self._say("  Sensor routing (XSensor/YSensor -> BNC): still in place")
        if self.cfg.z_signal.route:
            z = self.cfg.z_signal
            ok = self.z_ok()
            if ok is False:
                print(f"  Igor has reset the Z routing ({z.source} -> {z.bnc}) -- setting it again:")
                self.route_z()
            elif ok is True:
                self._say(f"  Z routing ({z.source} -> {z.bnc}): still in place")
        if getattr(self.cfg, "pfm", None) is not None and self.cfg.pfm.enabled:
            from . import pfm
            ok = pfm.routing_ok(self)
            if ok is False:
                print("  Igor has reset the PFM routing -- setting it again:")
                pfm.route(self)
            elif ok is True:
                self._say("  PFM routing (amplitude / phase -> BNC): still in place")

    # ------------------------------------------------------------------
    # X/Y sensors -> BNC outputs (-> PyScanner AI1/AI2), and their calibration
    # ------------------------------------------------------------------
    def _dac_free(self, dac: str, skip_rows=(), skip_loops=(), side: str = "Cypher") -> bool:
        rows = CYPHER_XPT_ALL if _side_name(side) == "Cypher" else ARC_XPT
        for row in rows:
            if row not in skip_rows and str(self.read_route(row, side)).casefold() == dac.casefold():
                return False
        for k in range(6):
            if k in skip_loops:
                continue
            try:
                g = self.read_loop(k, side=_side_name(side))
            except RuntimeError:
                continue
            if str(_g(g, "OutputChannel")).split(".")[-1].casefold() == dac[3:].casefold() \
                    and _running(g):
                return False
        return True

    def _copy_to_bnc(self, signal: str, k: int, bnc: str, gain: float, prefer=(), taken=(), side: str = "Cypher",
                     replaceable=()) -> str:
        """`signal` -> Cypher PIDS loop k (a straight copy: setpoint 0, P = gain, I = S = 0,
        always on) -> a free DAC (`prefer` first) -> `bnc`, locked. Returns the DAC. Refuses
        Igor's loops 0/1 and a loop running something else. WriteXPT is the caller's."""
        if k in (0, 1):
            raise ValueError(f"loop {k}: the Cypher's loops 0/1 are Igor's own (LVDT.X/Y -> Output.A/B)")
        sn = _side_name(side)
        rows, dacs, pop_prefix, lock_prefix = _side_tables(sn)
        g = self.read_loop(k, side=sn)
        cur_in = str(_g(g, "InputChannel")) if g is not None else ""
        # Igor reports the sensors by their channel names (seen 2026-11-09: a loop set to
        # XSensor reads back as LVDT.X): the same signal
        if _same_signal(cur_in, signal):
            cur_in = signal
        rep = [r for r in replaceable]
        if any(_same_signal(cur_in, r) for r in rep):
            cur_in = next(r for r in rep if _same_signal(cur_in, r))
        if g is not None and _running(g) and cur_in.casefold() != signal.casefold() \
                and cur_in.casefold() in {r.casefold() for r in replaceable}:
            # one of our own earlier copies (e.g. the topography notebook's ZSensor copy, still
            # running in Igor) that this run doesn't use: taken over
            self._say(f"  {sn} loop {k}: still running an earlier copy of {cur_in} "
                      f"({_g(g, 'OutputChannel')}) -- not used now, taken over for {signal}")
        elif g is not None and _running(g) and cur_in.casefold() != signal.casefold():
            raise RuntimeError(f"{sn}.PIDSLoop.{k} is in use ({_g(g, 'InputChannel')} -> "
                               f"{_g(g, 'OutputChannel')}): pick another loop")
        order = list(prefer) + [d for d in dacs if d not in prefer]
        if g is None:     # no Igor: show what would be sent
            dac = next(d for d in order if d not in taken)
        else:
            # the preferred DAC first; without one, keep what the BNC is on now if it's free
            cur = str(self.read_route(bnc, sn))
            cands = order if prefer else [cur] + [d for d in order if d.casefold() != cur.casefold()]
            dac = next((d for d in cands if d in dacs and d not in taken
                        and self._dac_free(d, skip_rows=(bnc,), skip_loops=(k,), side=sn)), None)
            if dac is None:
                raise RuntimeError(f"no free Cypher DAC for {signal} (tried {order})")
        g2 = dict(g or {})
        for key, val in (("InputChannel", signal), ("OutputChannel", "output." + dac[3:]), ("Setpoint", "0"),
                         ("DynamicSetpoint", "yes"), ("SetpointOffset", 0), ("dGain", 0),
                         ("pGain", f"{gain:g}"),
                         ("iGain", 0), ("sGain", 0), ("StartEvent", "Always"), ("StopEvent", "Never")):
            _put(g2, key, val)
        sc = -gain
        self._say(f"  {signal} -> {sn} loop {k} -> {dac} -> {bnc} (locked)"
                  + ("" if sc == 1 else f", divided by {1 / sc:g}" if abs(sc) < 1 else f", x {sc:g}"))
        # the loop starts from the DAC's present value (seen 2026-10-09: OutC still at ~10 V from
        # before -> stuck at the rail; with DynamicSetpoint the setpoint becomes the signal's
        # value at the start): preload the DAC with the copy's value first, as Oxford's
        # SensorFeedback example does
        if self.igor is not None:
            sv = self.read_value(signal)
            self.cmd(f'td_WriteValue("{sn}.Output.{dac[3:]}", {(-gain) * sv:.6f})')
        self.write_loop(k, g2, side=sn)
        idx = rows.index(bnc)
        self.ex(f"{pop_prefix}{bnc}Popup", PANEL, 0, dac)
        self.ex(f"{lock_prefix}{idx}Box_0", PANEL, 1, "")
        return dac

    def route_sensors(self) -> None:
        """XSensor -> Cypher loop x_loop -> a free DAC -> x_bnc (locked), Y likewise (_copy_to_bnc)."""
        ps = self.cfg.position_sensors
        taken = []
        for sensor, k, bnc in (("XSensor", ps.x_loop, ps.x_bnc), ("YSensor", ps.y_loop, ps.y_bnc)):
            taken.append(self._copy_to_bnc(sensor, k, bnc, ps.copy_gain, taken=taken))
        self.ex("WriteXPT", PANEL, 0, "")
        if self.igor is not None:          # check: each DAC is the sensor (/ divider) now
            time.sleep(0.2)
            bad = [s_ for s_, d_ in zip(("XSensor", "YSensor"), taken)
                   if not self._copy_matches(s_, d_[3:], ps.copy_gain, verbose=True)]
            if bad:
                print(f"  WARNING: {bad} not copied correctly -- the AI will not show the sensor position")

    # ------------------------------------------------------------------
    # the Z signal -> a BNC (-> PyScanner's readback AI): cfg.z_signal
    # ------------------------------------------------------------------
    def _xy_sensor_dacs(self) -> List[str]:
        """The DACs the X/Y sensor copies use (not available to Z)."""
        ps = self.cfg.position_sensors
        if not ps.enabled:
            return []
        if self.igor is None:
            return SENSOR_DACS[:2]
        out = []
        for k in (ps.x_loop, ps.y_loop):
            ch = str(_g(self.read_loop(k), "OutputChannel")).split(".")[-1]
            if ch:
                out.append("Out" + ch.upper())
        return out

    def route_z(self) -> None:
        """cfg.z_signal.source -> Cypher loop z.loop -> z.dac (or another free DAC) -> z.bnc
        (locked); then, tip withdrawn and at rest, checks that the readback AI reads that copy."""
        z = self.cfg.z_signal
        # the PFM notebook's copies (still running in Igor after a PFM session) may be taken over
        pf = getattr(self.cfg, "pfm", None)
        spare = [c.source for c in (pf.amplitude, pf.phase)] if pf is not None and not pf.enabled else []
        dac = self._copy_to_bnc(z.source, z.loop, z.bnc, z.copy_gain, prefer=(z.dac,), taken=self._xy_sensor_dacs(),
                                replaceable=spare)
        self.ex("WriteXPT", PANEL, 0, "")
        info = dict(source=z.source, loop=z.loop, dac=dac, bnc=z.bnc, divider=float(z.output_divider or 1.0),
                    ai_channel=self.cfg.readback.ai_channel, m_per_V=z.m_per_V, ai_matches=None)
        if z.m_per_V is None and z.source.casefold() == "zsensor" and self.igor is not None:
            try:
                m = self._num('GV("ZLVDTSens")')
                info["m_per_V"] = m if (m is not None and np.isfinite(m) and m > 0) else None
            except Exception:    # noqa: BLE001 -- unknown: heights stay in volts
                info["m_per_V"] = None
            if info["m_per_V"] is None:
                print('  NOTE: GV("ZLVDTSens") gave no sensitivity -- Z can\'t be converted to nm '
                      '(set cfg.z_signal.m_per_V, in m/V)')
        if self.igor is not None:
            time.sleep(0.2)
            self._copy_matches(z.source, dac[3:], z.copy_gain, verbose=True)
            info["ai_matches"] = self.check_z_ai(dac)
        if info["m_per_V"]:
            nm = info["m_per_V"] * info["divider"] * 1e9
            self._say(f"  {self.cfg.readback.ai_channel} -> length: {nm:.4g} nm per AI volt ({z.source} "
                      f"{info['m_per_V'] * 1e9:.4g} nm/V x divider {info['divider']:g}) -- saved with the run")
        self.z_info = info
        self.backend.z_signal_info = info          # every executor on this backend picks it up
        if self.executor is not None:
            self.executor.z_signal_info = info
            self.executor.set_z_units()

    def check_z_ai(self, dac: str) -> Optional[bool]:
        """At rest: does the readback AI read the copy on the DAC (= source / divider)? A
        wrong cable / BNC -> a warning (the run records whatever the AI is cabled to)."""
        z = self.cfg.z_signal
        n = max(int(0.2 * self.cfg.velocity.sample_rate_hz), 20)
        p = getattr(self.backend, "current_pos", None)
        x0, y0 = (float(p[0]), float(p[1])) if p is not None else (0.0, 0.0)
        ai = np.asarray(self.backend.measure_path(np.full(n, x0), np.full(n, y0)), dtype=float)
        dv = self.read_value(f"Cypher.Output.{dac[3:]}")
        sv = self.read_value(z.source)
        if not len(ai) or dv is None:
            return None
        a = float(np.mean(ai))
        ok = abs(a - dv) <= z.check_tol_V
        self._say(f"  {self.cfg.readback.ai_channel} reads {a:+.4f} V (std {np.std(ai) * 1e3:.2f} mV); {dac} = "
                  f"{dv:+.4f} V ({z.source} {sv:+.4f} V / {z.output_divider:g})"
                  + (f"  -> the readback is {z.source}" if ok else ""))
        if not ok:
            print(f"  WARNING: {self.cfg.readback.ai_channel} does NOT read the {z.source} copy -- is its cable "
                  f"on {z.bnc}? (the run records whatever {self.cfg.readback.ai_channel} is cabled to)")
        return ok

    def z_ok(self) -> Optional[bool]:
        """The Z copy still in place (loop running on the source, the BNC on its DAC, a copy)?"""
        if self.igor is None:
            return None
        z = self.cfg.z_signal
        g = self.read_loop(z.loop)
        out = str(_g(g, "OutputChannel")).split(".")[-1]
        if not _same_signal(_g(g, "InputChannel"), z.source) \
                or not _active(g) \
                or str(self.read_route(z.bnc)).casefold() != f"out{out}".casefold() \
                or not _p_ok(g, z.copy_gain)                 or not _setpoint_zero(g):
            return False
        return self._copy_matches(z.source, out, z.copy_gain)

    def calibrate_sensors(self) -> None:
        """Tip withdrawn: the sensor calibration (sensors.calibrate) -> executor.sensor_cal."""
        from . import sensors
        read = (lambda name: self.read_value(name)) if self.igor is not None else None
        gv = (lambda name: self._num(f'GV("{name}")')) if self.igor is not None else None
        cal = sensors.calibrate(self.backend, self.cfg, read_sensor_V=read, igor_gv=gv, verbose=self.verbose)
        self.sensor_cal = cal
        self.backend.sensor_cal = cal      # every executor on this backend picks it up (reference raster)
        if self.executor is not None:
            self.executor.sensor_cal = cal
            self.executor.last_pos = np.zeros(2)

    # ------------------------------------------------------------------
    # the sequences
    # ------------------------------------------------------------------
    def setup(self, withdraw_first: bool = True) -> None:
        """Before any scan: (withdraw), centre, AO0/AO1 at 0, crosspoint + X/Y loops, engage."""
        t0 = time.time()
        print("=" * 72)
        print("Asylum setup" + ("" if self.igor is not None else " -- NO IGOR: commands printed, not sent"))
        print("=" * 72)
        if withdraw_first:
            print("0. Withdraw first (nothing moves laterally while engaged)")
            self.withdraw()
        print("1. Tip to the centre of the image")
        if self.cfg.asylum.center_first:
            self.center()
        else:
            print("  skipped (cfg.asylum.center_first = False)")
        print("2. FPGA outputs AO0/AO1 at 0")
        self.zero_ao()
        print("3. Crosspoint and X/Y loops")
        self.set_crosspoint()
        self.route_xy_to_fpga()
        if self.cfg.position_sensors.enabled:
            print("3b. X/Y position sensors -> BNC outputs (-> PyScanner "
                  f"{self.cfg.position_sensors.x_ai_channel}/{self.cfg.position_sensors.y_ai_channel})")
            self.route_sensors()
            print("3c. Sensor calibration (tip withdrawn)")
            self.calibrate_sensors()
        if self.cfg.z_signal.route:
            print(f"3d. Z signal: {self.cfg.z_signal.source} -> {self.cfg.z_signal.bnc} "
                  f"(-> PyScanner {self.cfg.readback.ai_channel})")
            self.route_z()
        if getattr(self.cfg, "pfm", None) is not None and self.cfg.pfm.enabled:
            from . import pfm
            names = ", ".join(f"{c.label} ({c.source}) -> {c.side} {c.bnc} -> {c.ai_channel}"
                              for c in self.cfg.pfm.channels().values())
            print(f"3e. PFM signals: {names}")
            pfm.route(self)
        print("4. Engage")
        self.engage()
        self.setup_done = True
        print(f"Setup done in {time.time() - t0:.1f} s -- the tip is engaged at (0, 0)")

    def finish(self) -> None:
        """Withdraw, then the FPGA outputs smoothly back to (0, 0)."""
        print("-" * 72)
        print("Finishing: withdraw, then back to (0, 0)")
        try:
            self.withdraw()
        finally:
            self.zero_ao()
        print("-" * 72)

    @contextlib.contextmanager
    def run(self, label: str = "scan", executor=None):
        """Engaged for the block (engages if withdrawn); withdrawn and back at (0, 0)
        afterwards -- also when the block fails or is interrupted."""
        if not self.setup_done:
            print("NOTE: setup() hasn't run in this session -- assuming the crosspoint and X/Y "
                  "loops are already set (run the setup cell first if not)")
        if executor is not None:      # e.g. the session was rebuilt after the setup
            self.executor = executor
        if self.executor is not None and self.executor.sensor_cal is None and self.sensor_cal is not None:
            self.executor.sensor_cal = self.sensor_cal    # the setup's calibration, on this executor too
        if self.executor is not None and self.z_info is not None:
            self.executor.z_signal_info = self.z_info
            self.executor.set_z_units()
        if self.cfg.position_sensors.enabled and self.executor is not None and self.executor.sensor_cal is None:
            print("NOTE: position sensors on but not calibrated in this session (setup cell) -- the run "
                  "records them but uses the commanded positions")
        if not self.engaged:
            print(f"Checking the routing before: {label}")
            self.ensure_routing()
            print(f"Engaging for: {label}")
            self.engage()
        try:
            yield self
        except KeyboardInterrupt:
            print(f"\n{label} interrupted")
            raise
        finally:
            self.finish()


_LOOP_KEYS = {"inputchannel", "outputchannel", "setpoint", "pgain", "igain", "sgain", "startevent", "stopevent"}


def _g(group, key, default=""):
    """Case-insensitive lookup in a td_ReadGroup dict."""
    for a, b in (group or {}).items():
        if a.casefold() == key.casefold():
            return b
    return default


def _same_signal(a, b) -> bool:
    """Igor reports a loop's sensor input by its channel name (XSensor reads back as LVDT.X,
    seen 2026-11-09): the same signal either way."""
    al = {"lvdt.x": "xsensor", "lvdt.y": "ysensor", "lvdt.z": "zsensor",
          "lockina.0.r": "amplitude", "lockina.0.theta": "phase"}
    a, b = str(a).casefold(), str(b).casefold()
    return al.get(a, a) == al.get(b, b)


def _running(group) -> bool:
    """The loop runs: its StartEvent isn't Never / empty. Igor reports it with a trailing
    ';' ("Never;", "Always;" -- seen 2026-10-13), so separators and spaces are dropped."""
    return str(_g(group, "StartEvent")).strip().strip(";").strip().casefold() not in ("", "never")


def _active(group) -> bool:
    """The copy loop is set to run AND is running: Igor can stop a loop (Status "Stopped") and
    leave its StartEvent "Always" -- seen 2026-11-04 after a withdraw: the PFM copies' outputs then
    stayed frozen at their last value through a whole raster."""
    return _running(group) and "stop" not in str(_g(group, "Status")).casefold()


def _setpoint_zero(group, tol: float = 1e-3) -> bool:
    """The copy loop's setpoint is 0 (seen 2026-11-02: after a withdraw / an Igor scan it was the
    sensor's value at that moment, e.g. "-1.4204276 V" -- the copy then reads ~0 wherever the tip is)."""
    try:
        return abs(float(str(_g(group, "Setpoint")).split()[0])) < tol
    except (ValueError, IndexError):
        return False


def _p_ok(group, gain: float) -> bool:
    """The copy loop's P gain is the configured one."""
    try:
        return abs(float(_g(group, "pGain")) - gain) < 1e-6
    except ValueError:
        return False


def _put(group, key, value) -> None:
    for a in list(group):
        if a.casefold() == key.casefold():
            group[a] = value
            return
    group[key] = value
