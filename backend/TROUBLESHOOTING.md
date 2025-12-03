# Troubleshooting Guide

## Issue: ModuleNotFoundError: No module named 'mcp.server'

### Error Message
```
ModuleNotFoundError: No module named 'mcp.server'
```

### Root Cause
Two issues were identified:

1. **Namespace Collision**: The local `backend/mcp/` directory was shadowing the installed `mcp` package from PyPI
2. **Incorrect Import Path**: Code was using outdated import path `from mcp.server.fastmcp import FastMCP`
3. **Incorrect API Method**: Code was calling `mcp.get_asgi_app()` instead of `mcp.sse_app`

### Resolution Steps

#### 1. Renamed Local Directory
```bash
# Renamed backend/mcp/ to backend/mcp_server/ to avoid namespace collision
mv backend/mcp backend/mcp_server
```

#### 2. Updated Import Statements
**File: `backend/mcp_server/tools.py`**
```python
# Before:
from mcp.server.fastmcp import FastMCP

# After:
from mcp.server import FastMCP
```

**File: `backend/main.py`**
```python
# Before:
from mcp.tools import mcp, initialize_tools

# After:
from mcp_server.tools import mcp, initialize_tools
```

#### 3. Fixed API Method Call
**File: `backend/main.py`**
```python
# Before:
app.mount("/mcp", mcp.get_asgi_app())

# After:
app.mount("/mcp", mcp.sse_app())  # Note: sse_app() is a method, not a property
```

### Verification

Test the application imports successfully:

```bash
cd /Users/attila.pogany/Code/projects/librarian-mcp
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'backend')
from main import app
print('✅ Application imported successfully')
"
```

### Dependencies Status

The `backend/requirements.txt` is **correct** and complete. No changes needed to requirements.

Current MCP dependency:
```
mcp[cli]>=1.12.3,<2.0
```

### Notes

- The `mcp` package version 1.12.3+ uses `from mcp.server import FastMCP` as the correct import path
- Local directory names should not conflict with installed package names to avoid shadowing
- FastMCP provides `sse_app()` **method** (not property) that returns the SSE ASGI app
- The SSE endpoint will be available at the mount path + `/sse` (e.g., `/mcp/sse`)

### Related Files Changed

1. `backend/mcp/` → `backend/mcp_server/` (directory renamed)
2. `backend/mcp_server/tools.py` (import updated)
3. `backend/main.py` (import updated, API call fixed)
