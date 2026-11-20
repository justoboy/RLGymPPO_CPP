cmake_minimum_required (VERSION 3.8)

project("RLGymPPO_CPP")

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Make sure CMake finds libtorch if its in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message("Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "libtorch")
	# Make ultra-sure we can find libtorch if its local
	set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files. The pybind11 module bindings are EXCLUDED here — they define a
# PYBIND11_MODULE entry point and are compiled into their own extension-module target below, not
# into this library.
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")
list(FILTER FILES_SRC EXCLUDE REGEX ".*Bindings/.*")

add_library(RLGymPPO_CPP SHARED ${FILES_SRC})
target_compile_definitions(RLGymPPO_CPP PRIVATE -DWITHIN_RLGPC)
target_include_directories(RLGymPPO_CPP PUBLIC "src/public")
target_include_directories(RLGymPPO_CPP PRIVATE "src/private")

# Include libtorch (PUBLIC so consumers that include our public headers, which expose
# torch::nn::Module types, also get the Torch include dirs on their compile line)
target_link_libraries(RLGymPPO_CPP PUBLIC "${TORCH_LIBRARIES}")

if (TORCH_CUDA_LIBRARIES)
	message("Enabling CUDA support...")
	target_compile_definitions(RLGymPPO_CPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(RLGymPPO_CPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymPPO_CPP PROPERTIES CXX_STANDARD 20)

# Include RLGymSim_CPP
add_subdirectory(RLGymSim_CPP)
target_link_libraries(RLGymPPO_CPP PUBLIC RLGymSim_CPP)

# Model-side parity gate executable: compiles the model sources directly (no DLL import macros) and
# links the static sim + Torch, so it can be run standalone to validate C++/Python parity of the
# actor/FSP forward. Named ModelParityGate to avoid colliding with the sim's obs-builder ParityGate.
add_executable(ModelParityGate
	"${PROJECT_SOURCE_DIR}/../tools/parity_main.cpp"
	"src/private/RLGymPPO_CPP/Model/EARLPerceiver.cpp"
	"src/private/RLGymPPO_CPP/Model/PerceiverNet.cpp"
	"src/private/RLGymPPO_CPP/Model/FSPNet.cpp"
	"src/private/RLGymPPO_CPP/PPO/GaussianPolicy.cpp")
target_include_directories(ModelParityGate PRIVATE "src/public" "src/private")
target_link_libraries(ModelParityGate PUBLIC "${TORCH_LIBRARIES}" RLGymSim_CPP)
set_target_properties(ModelParityGate PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

# League gradient-direction parity gate (P3): loads the BC actor, runs the exact PPO policy-loss
# backward, and dumps per-parameter gradients for the Python side to diff. Named GradGate.
add_executable(GradGate
	"${PROJECT_SOURCE_DIR}/../tools/grad_main.cpp"
	"src/private/RLGymPPO_CPP/Model/EARLPerceiver.cpp"
	"src/private/RLGymPPO_CPP/Model/PerceiverNet.cpp"
	"src/private/RLGymPPO_CPP/Model/FSPNet.cpp"
	"src/private/RLGymPPO_CPP/PPO/GaussianPolicy.cpp")
target_include_directories(GradGate PRIVATE "src/public" "src/private")
target_link_libraries(GradGate PUBLIC "${TORCH_LIBRARIES}" RLGymSim_CPP)
set_target_properties(GradGate PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

# Isolation test: does the live Gym env Reset() work in this environment (no torch/pybind/Learner)?
add_executable(isolation_env "${PROJECT_SOURCE_DIR}/../tools/isolation_env.cpp")
target_link_libraries(isolation_env PUBLIC RLGymSim_CPP)
set_target_properties(isolation_env PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

# Standalone microbenchmark for the obs->tensor conversion hot path (Tier A candidate).
add_executable(obsbench "${PROJECT_SOURCE_DIR}/../tools/obsbench.cpp")
target_link_libraries(obsbench PUBLIC "${TORCH_LIBRARIES}")
set_target_properties(obsbench PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

# Include JSON
#target_include_directories(RLGymPPO_CPP PRIVATE "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(RLGymPPO_CPP PUBLIC ${PYTHON_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(RLGymPPO_CPP PUBLIC pybind11::embed)

# Python extension module exposing the single-policy learner to Python. Links the library; the
# Python driver imports this module and composes N learners into a league in Python.
pybind11_add_module(rlgym_learner "src/private/RLGymPPO_CPP/Bindings/LearnerBindings.cpp")
target_include_directories(rlgym_learner PRIVATE "src/public" "src/private")
target_link_libraries(rlgym_learner PRIVATE RLGymPPO_CPP)
set_target_properties(rlgym_learner PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message("Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)
endif (MSVC)