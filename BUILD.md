# Build & Environment Setup

This project builds with **CMake + MSVC (Visual Studio 2022) + libtorch 2.2 (CPU)**.
The recurring build failures ("cmake not found", "torch.dll not found", "cl.exe not
found") all come from tools/DLLs not being on `PATH` in a fresh terminal. This doc
fixes that permanently.

---

## TL;DR — the one command to remember

In **every new terminal** before building, run:

```cmd
call setup_env.cmd
```

Then build:

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Permanent fix (recommended — do this once, forget it forever)

Instead of running `setup_env.cmd` every time, add these to your **permanent user
PATH** so every new terminal already has them:

1. Press `Win`, type **"env"**, click **"Edit the system environment variables"**.
2. Click **"Environment Variables…"**.
3. Under **"User variables for justo"**, select `Path` → **Edit**.
4. **New** and paste each of these lines:

   ```
   C:\Program Files\JetBrains\CLion 2025.3.6.2\bin\cmake\win\x64\bin
   C:\Users\justo\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch\lib
   C:\Python314
   C:\Python314\Scripts
   ```

5. Also add a **New** user variable:
   - Name: `CMAKE_PREFIX_PATH`
   - Value: `C:\Users\justo\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch`

6. OK out, then **restart VS Code / your terminal** (PATH changes don't apply to
   already-open terminals).

After that, `cmake --version` works in any new terminal and the built `.exe`s find
`torch.dll`/`c10.dll` at runtime with no extra steps.

---

## What each path is for

| Path | Why it's needed |
|------|-----------------|
| `...\CLion 2025.3.6.2\bin\cmake\win\x64\bin` | There is **no** system-wide `cmake`. This is CLion's bundled `cmake.exe`. |
| `...\pytorch-2.2.0-cpu_...\torch\lib` | Runtime DLLs (`torch.dll`, `c10.dll`, `torch_cpu.dll`, `fbgemm.dll`, …). Without this, `ParityGate.exe` / `RLGymPPO_CPP_Example.exe` crash on launch with "c10.dll not found". |
| `C:\Python314` + `\Scripts` | Python 3.14 for `tools/convert_bc_checkpoint.py` and pybind11 embed. |
| `CMAKE_PREFIX_PATH` (env var) | Lets `find_package(Torch REQUIRED)` locate the libtorch CMake config without passing `-DCMAKE_PREFIX_PATH` by hand. |

> **MSVC / `cl.exe`** is provided automatically by the Visual Studio generator —
> CMake calls `vcvars64.bat` for you, so you do **not** need to add the VS compiler
> to PATH manually.

---

## Common errors → cause → fix

| Symptom | Cause | Fix |
|---------|-------|-----|
| `'cmake' is not recognized…` | CLion's cmake not on PATH | `call setup_env.cmd` or add path (above) |
| `c10.dll was not found` / `torch.dll not found` on run | libtorch `lib` not on PATH | add `...\torch\lib` to PATH |
| `Could not find a package configuration file provided by "Torch"` | `CMAKE_PREFIX_PATH` unset | set the env var, or pass `-DCMAKE_PREFIX_PATH=...` |
| New `.cpp` not compiled / unresolved externals | stale `GLOB_RECURSE` cache | `rmdir /s /q build` then reconfigure (GLOB is snapshot-based) |

---

## Full clean build (copy-paste)

```cmd
call setup_env.cmd
rmdir /s /q build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

> Always `rmdir /s /q build` when you **add or rename a source file** — the
> `CMakeLists.txt` uses `GLOB_RECURSE`, which only re-scans at configure time.
