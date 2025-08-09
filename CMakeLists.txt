cmake_minimum_required(VERSION 3.10)
project(release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Static libraries
set(GLFW_BIN      ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw-3.4.bin.MACOS/lib-arm64/libglfw3.a)
set(GLEW_BIN      ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glew-2.2.0/lib/libGLEW.a)
set(FREETYPE_BIN  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/freetype/build/libfreetype.a)

# Optional: Add harfbuzz if freetype depends on it
# set(HARFBUZZ_BIN  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/harfbuzz/build/libharfbuzz.a)
set(BZ2_BIN       /opt/homebrew/lib/libbz2.a)
set(ZLIB_BIN      /opt/homebrew/lib/libz.a)
set(PNG_BIN       /opt/homebrew/lib/libpng16.a)
set(BROTLI_BIN    /opt/homebrew/lib/libbrotlidec.a)

# Include directories
set(LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(
    ${LOCAL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw-3.4.bin.MACOS/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glew-2.2.0/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/freetype/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    # ${CMAKE_CURRENT_SOURCE_DIR}/vendor/harfbuzz/include
    /opt/homebrew/include
)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(release ${SOURCES} "main.cpp")

# Link everything together
target_link_libraries(release
    ${GLFW_BIN}
    ${GLEW_BIN}
    ${FREETYPE_BIN}
    # ${HARFBUZZ_BIN}
    # ${BZ2_BIN}
    # ${ZLIB_BIN}
    # ${PNG_BIN}
    # ${BROTLI_BIN}
    "-framework Cocoa"
    "-framework OpenGL"
    "-framework IOKit"
    "-framework CoreVideo"
    "-framework QuartzCore"
)
