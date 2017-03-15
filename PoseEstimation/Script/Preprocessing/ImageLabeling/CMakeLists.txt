cmake_minimum_required(VERSION 3.3)
project(ImageLabeling)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
    main.cpp
    opening.cpp
    labeling.cpp)

add_executable(ImageLabeling ${SOURCE_FILES})
target_link_libraries(ImageLabeling ${Boost_LIBRARIES})
target_link_libraries(ImageLabeling ${OpenCV_LIBRARIES})
