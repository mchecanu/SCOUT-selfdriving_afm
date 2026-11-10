"""Launch the selfdriving_afm MCP server from this folder (no install needed).

    python scout_mcp.py                 # stdio: what Claude Desktop / Claude Code start
    python scout_mcp.py --no-load-config

The working folder is this one (notebooks, hardware_paths.json, scan_data/). See
README "MCP server" for the client configuration.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from selfdriving_afm.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main(["--workdir", str(pathlib.Path(__file__).resolve().parent)] + sys.argv[1:])
