# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
#cmake_minimum_required (VERSION 2.8.11)
cmake_minimum_required(VERSION 3.1)
project (stegerMatch)

find_package( OpenCV REQUIRED )
add_executable( stegerMatch StegerMatch.cpp )
target_link_libraries( stegerMatch ${OpenCV_LIBS} )



# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
#add_subdirectory (Hello)
#add_subdirectory (Demo)