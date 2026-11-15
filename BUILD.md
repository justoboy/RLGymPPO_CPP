# Build & Environment Setup (permanent, machine-wide)

This project builds with **CMake + MSVC (Visual Studio 2022) + libtorch 2.4 (CUDA 11.8, sm_86)**.
The recurring build failures ("cmake not found", "c10.dll/torch.dll not found")
happen because `cmake.exe` and the libtorch runtime DLLs are not on your **PATH**.

> **CRITICAL — do NOT let the build fall back to the CPU libtorch.** The learner MUST link the
> **CUDA** libtorch (`torch 2.4.0+cu11.8`), not the CPU `pytorch-2.2.0-cpu_py39` package. A stale
> `CMAKE_PREFIX_PATH` pointing at the CPU `pkgs\pytorch-2.2.0-cpu_*` path silently reverts the learner
> to CPU compute, which destroys the C++/Python speed parity (Python runs on CUDA). The correct value
> is the **installed** CUDA torch: `C:\Users\justo\anaconda3\lib\site-packages\torch` (NOT a `pkgs`
> path). After changing it, `rmdir /s /q build` and reconfigure — the configure log MUST print
> `Enabling CUDA support...` and list `torch_cuda.dll`/`c10_cuda.dll` in TORCH_DLLS.

This is a **machine-level** setting, not a per-project one — fix it once and every
project (reference, RLGymSim_CPP, rlgym, this one…) stops hitting it. Do **not** commit
a per-project `setup_env.cmd`; that is machine-specific and wrong to version-control.

---

## What needs to be on PATH (and why)

| What | Example location on this machine | Why |
|------|-------------------------------|-----|
| `cmake.exe` | `%LOCALAPPDATA%\JetBrains\...` or `C:\Program Files\JetBrains\CLion*\bin\cmake\win\x64\bin` | No system-wide cmake; CLion bundles one. |
| libtorch `lib` DLLs | `%USERPROFILE%\anaconda3\pkgs\pytorch-2.2.0-cpu_*\Lib\site-packages\torch\lib` | Runtime `torch.dll`/`c10.dll`/`torch_cpu.dll` for the built `.exe`s. |
| Python | `C:\Python314`, `C:\Python314\Scripts` | `tools/convert_bc_checkpoint.py` + pybind11 embed. |
| `CMAKE_PREFIX_PATH` (var) | `...\site-packages\torch` | Lets `find_package(Torch REQUIRED)` find libtorch. |

> **MSVC / `cl.exe` is NOT on PATH and must not be.** The Visual Studio CMake generator
> calls `vcvars64.bat` for you. Adding the VS compiler to PATH manually is a common
> mistake that causes subtle toolchain conflicts.

> Find your own paths: `where.exe python` for Python; the torch lib dir is the
> `...\pkgs\pytorch-*\Lib\site-packages\torch\lib` folder from your conda env; the cmake
> dir is your IDE's bundled `bin\cmake\win\x64\bin`.

---

## Permanent fix — Option A: GUI (recommended, no truncation risk)

1. `Win` → type **"env"** → **Edit the system environment variables** → **Environment Variables…**
2. Under **User variables**, select `Path` → **Edit** → **New**, add each path from the table above.
3. Under **User variables** → **New**: name `CMAKE_PREFIX_PATH`, value = your `...\site-packages\torch` dir.
4. OK → **restart VS Code / your terminal** (PATH changes don't apply to already-open terminals).

## Permanent fix — Option B: `setx` (one-shot, persistent)

`setx` writes to the registry (HKCU\Environment) and survives reboots. Run in **cmd**:

```cmd
setx CMAKE_PREFIX_PATH "C:\Users\%USERNAME%\anaconda3\lib\site-packages\torch"
```

> Note the path: it is the **installed** CUDA torch at `anaconda3\lib\site-packages\torch` (torch
> 2.4.0+cu11.8), NOT the old `pkgs\pytorch-2.2.0-cpu_*` CPU package. Pointing it at the CPU `pkgs`
> path is exactly what silently reverts the learner to CPU compute.

For PATH, **do not** do `setx PATH "%PATH%;..."` — `%PATH%` is the *merged* machine+user
value and `setx` truncates at 1024 chars, silently corrupting your PATH. Instead use the
GUI (Option A) for PATH, or the safe read-modify-write below.

### Safe PATH append (PowerShell, idempotent — only appends missing entries)

```powershell
$dirs = @(
  'C:\Program Files\JetBrains\CLion 2025.3.6.2\bin\cmake\win\x64\bin',
  "$env:USERPROFILE\anaconda3\lib\site-packages\torch\lib",
  'C:\Python314',
  'C:\Python314\Scripts'
)
$cur  = [Environment]::GetEnvironmentVariable('Path','User'); if (-not $cur) { $cur = '' }
$parts = $cur -split ';'
foreach ($d in $dirs) { if ($parts -notcontains $d) { $cur = if ($cur) { "$cur;$d" } else { $d } } }
[Environment]::SetEnvironmentVariable('Path', $cur, 'User')
```

---

## Verify (after restarting the terminal)

```cmd
where cmake
where python
echo %CMAKE_PREFIX_PATH%
```

All three should resolve to the paths above.

## Build

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

> `rmdir /s /q build` and re-configure whenever you **add/rename a source file** —
> `CMakeLists.txt` uses `GLOB_RECURSE`, which only re-scans at configure time.

### CRITICAL: Python must be 3.9 (anaconda), not 3.14

The C++ library links a **Python 3.9** libtorch (`pytorch-2.2.0-cpu_py39`). If `find_package`
picks up `C:\Python314` (3.14), the built `RLGymPPO_CPP.dll` and the `rlgym_learner` extension
module link the wrong `python3xx.lib` and fail to import (`DLL load failed` / ABI mismatch). Pin
Python to anaconda 3.9 at configure time. Because the library uses the **old** `find_package(PythonLibs)`
(it reads `PYTHON_*`, not `Python3_*`), you must pass the old-style cache vars too:

Pass `-DCMAKE_PREFIX_PATH` explicitly so the configure does not depend on a (possibly stale) env var:

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_PREFIX_PATH=C:/Users/justo/anaconda3/lib/site-packages/torch ^
  -DPYTHON_EXECUTABLE=C:/Users/justo/anaconda3/python.exe ^
  -DPYTHON_INCLUDE_DIR=C:/Users/justo/anaconda3/include ^
  -DPYTHON_LIBRARY=C:/Users/justo/anaconda3/libs/python39.lib ^
  -DPython_EXECUTABLE=C:/Users/justo/anaconda3/python.exe
```

Verify the configure log prints `Enabling CUDA support...`, `Found CUDA`, and lists `torch_cuda.dll`
/`c10_cuda.dll` in TORCH_DLLS, plus `Found PythonLibs: .../python39.lib`. The module then builds as
`rlgym_learner.cp39-win_amd64.pyd`.

To import the module from Python, **import `torch` FIRST** (so the C10 dispatcher is initialized before
the extension's static initializers run — otherwise you get `Key already registered with the same
priority: C10`), then add the build/Release dir to the DLL search path and `sys.path`:

```python
import torch                      # MUST be first: initializes the C10 dispatcher before the .pyd loads
import os, sys
d = r"D:\RLBotTraining\RLGymPPO_CPP\build\Release"
os.add_dll_directory(d)            # RLGymPPO_CPP.dll + torch CUDA DLLs
sys.path.insert(0, d)              # so `import rlgym_learner` finds the .pyd
import rlgym_learner as m
```

## Revert

User PATH / `CMAKE_PREFIX_PATH` live in **HKCU\Environment**. To undo, use the same
Environment Variables GUI and delete the entries, or:
```powershell
[Environment]::SetEnvironmentVariable('CMAKE_PREFIX_PATH',$null,'User')
```
