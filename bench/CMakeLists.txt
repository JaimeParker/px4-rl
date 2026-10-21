add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE px4rl::cuda CUDA::cudart_static)
add_test(NAME bench COMMAND bench)
set_tests_properties(bench PROPERTIES LABELS bench)
