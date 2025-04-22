include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.1  # Use the latest release tag
)

FetchContent_MakeAvailable(Catch2)

# Make Catch2 CMake modules available
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras) 