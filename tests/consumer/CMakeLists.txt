cmake_minimum_required(VERSION 3.18)
project(px4rl_consumer LANGUAGES CXX)

find_package(px4rl 0.1 REQUIRED)

add_executable(px4rl_consumer main.cpp)
target_link_libraries(px4rl_consumer PRIVATE px4rl::px4)
