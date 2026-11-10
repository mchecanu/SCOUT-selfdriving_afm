"""Check the MCP server on this PC -- no hardware is touched.

    python mcp_selftest.py

Starts scout_mcp.py over stdio (as Claude Desktop / Claude Code do), lists the tools, runs a
short gp_path simulation and a DRY-RUN acquisition (simulated backend, Igor commands only
printed): setup, priors, a 10 V gp_path scan, its equivalent raster, status, park, scan summary
and a TV reconstruction; then the stepwise grid_refine in simulation. About 2-4 min.
"""
import asyncio
import json
import pathlib
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = pathlib.Path(__file__).resolve().parent


def _json(res):
    t = [c.text for c in res.content if getattr(c, "type", "") == "text"][0]
    if t.startswith("ERROR"):
        raise RuntimeError(t)
    return json.loads(t)


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "scout_mcp.py")])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print(f"OK  {len(tools)} tools: {', '.join(tools)}")
            info = _json(await s.call_tool("server_info", {}))
            print(f"OK  config from {info['config_source']} -- mode {info['mode']}, GP preset {info['gp_preset']}")
            await s.call_tool("set_sample", {"preset": "three_shapes"})
            v = _json(await s.call_tool("simulate", {"mode": "gp_path", "path_budget_V": 10, "wait_s": 240}))
            e = v.get("result", {}).get("evaluation", {})
            print(f"OK  simulate gp_path 10 V: {v['status']}, bad px {e.get('bad_px')}, GP compute "
                  f"{v.get('result', {}).get('gp_compute_per_segment_s')}")
            _json(await s.call_tool("connect_instrument", {"dry_run": True}))
            refused = [c.text for c in (await s.call_tool("run_scan", {})).content][0]
            print("OK  run_scan without confirm is refused:", refused.startswith("ERROR: PermissionError"))
            _json(await s.call_tool("microscope_setup", {"confirm": True, "wait_s": 120}))
            _json(await s.call_tool("set_priors", {"feature_min": 0.5, "feature_max": 2.0, "area_half_width": 4.0}))
            v = _json(await s.call_tool("run_scan", {"confirm": True, "mode": "gp_path", "path_budget_V": 10, "wait_s": 5}))
            v = _json(await s.call_tool("job_status", {"job_id": v["job_id"], "wait_s": 240}))
            res = v.get("result", {})
            print(f"OK  dry-run scan: {v['status']}, {res.get('waveforms')} waveforms, saved {res.get('saved')}")
            v = _json(await s.call_tool("acquire_reference_raster", {"confirm": True, "wait_s": 240}))
            print(f"OK  equivalent raster (user-defined): {v['status']}")
            _json(await s.call_tool("withdraw_and_park", {"confirm": True}))
            sm = _json(await s.call_tool("scan_summary", {}))
            print(f"OK  scan_summary: {sm.get('mode')}, {sm.get('waveforms')} waveforms, ~{sm.get('est_instrument_s')} s "
                  f"on the instrument, GP {sm.get('gp_path', {}).get('compute_per_segment_s')}")
            v = _json(await s.call_tool("reconstruct_scan", {"method": "tv"}))
            print(f"OK  reconstruct_scan tv: rows {[r['name'] for r in v['rows']]}")
            await s.call_tool("set_sample", {"preset": "three_shapes"})
            for a in ("setup", "level0", "status"):
                v = _json(await s.call_tool("stepwise_grid_refine", {"action": a, "simulate": True, "wait_s": 240}))
            print(f"OK  stepwise_grid_refine (simulated): done {v.get('done')}")
            print("All good -- add the server to your MCP client (README 'MCP server').")


if __name__ == "__main__":
    asyncio.run(main())
