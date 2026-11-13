@echo off
REM ============================================================================
REM  setup_env.cmd  -  one-shot environment setup for RLGymPPO_CPP builds
REM ----------------------------------------------------------------------------
REM  Run this in EVERY new terminal before building, OR add the PATH lines to
REM  your permanent user PATH (see BUILD.md "Permanent setup" section) so you
REM  never have to think about it again.
REM
REM  Usage:  call setup_env.cmd
REM  (use `call` so the PATH changes persist in YOUR shell, not just a child)
REM ============================================================================

REM --- 1. cmake (CLion's bundled cmake.exe; no system-wide cmake install) ----
set "CLION_CMAKE=C:\Program Files\JetBrains\CLion 2025.3.6.2\bin\cmake\win\x64\bin"
set "PATH=%CLION_CMAKE%;%PATH%"

REM --- 2. libtorch runtime DLLs ----------------------------------------------
REM  This is the #1 source of "torch.dll / c10.dll not found" crashes when you
REM  run ParityGate.exe or RLGymPPO_CPP_Example.exe. The CPU-only libtorch lives
REM  in the anaconda pkgs cache; its DLLs must be on PATH at RUNTIME.
set "TORCH_LIB=C:\Users\justo\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"

REM --- 3. Python (for tools/convert_bc_checkpoint.py and pybind11 embed) ------
set "PATH=C:\Python314;C:\Python314\Scripts;%PATH%"

REM --- 4. Export CMAKE_PREFIX_PATH so find_package(Torch) just works ----------
set "CMAKE_PREFIX_PATH=C:\Users\justo\anaconda3\pkgs\pytorch-2.2.0-cpu_py39hb0bdfb8_0\Lib\site-packages\torch"

echo.
echo [setup_env] PATH configured for this terminal session.
echo [setup_env]   cmake  : %CLION_CMAKE%
echo [setup_env]   torch  : %TORCH_LIB%
echo [setup_env] Verify with:  cmake --version
echo.
