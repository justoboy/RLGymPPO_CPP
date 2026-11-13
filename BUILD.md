# Build & Environment Setup (permanent, machine-wide)

This project builds with **CMake + MSVC (Visual Studio 2022) + libtorch 2.2 (CPU)**.
The recurring build failures ("cmake not found", "c10.dll/torch.dll not found")
happen because `cmake.exe` and the libtorch runtime DLLs are not on your **PATH**.

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
setx CMAKE_PREFIX_PATH "C:\Users\%USERNAME%\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch"
```

For PATH, **do not** do `setx PATH "%PATH%;..."` — `%PATH%` is the *merged* machine+user
value and `setx` truncates at 1024 chars, silently corrupting your PATH. Instead use the
GUI (Option A) for PATH, or the safe read-modify-write below.

### Safe PATH append (PowerShell, idempotent — only appends missing entries)

```powershell
$dirs = @(
  'C:\Program Files\JetBrains\CLion 2025.3.6.2\bin\cmake\win\x64\bin',
  "$env:USERPROFILE\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch\lib",
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

## Revert

User PATH / `CMAKE_PREFIX_PATH` live in **HKCU\Environment**. To undo, use the same
Environment Variables GUI and delete the entries, or:
```powershell
[Environment]::SetEnvironmentVariable('CMAKE_PREFIX_PATH',$null,'User')
```
