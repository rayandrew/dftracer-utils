# ##############################################################################
# DataFrame Library Tests
# ##############################################################################

set(DATAFRAME_TEST_SOURCES test_mask.cpp test_series.cpp test_result_batch.cpp test_methods.cpp
                           test_arrow.cpp test_field.cpp test_op_registry.cpp test_lazyframe.cpp
                           test_spill.cpp test_agg_state.cpp)

foreach(test_file ${DATAFRAME_TEST_SOURCES})
  string(REPLACE ".cpp" "" bin_exec ${test_file})
  string(REPLACE "/" "_" target_name "dataframe_${bin_exec}")

  add_executable(${target_name} ${test_file})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${target_name} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  # Some tests include internal SIMD kernel headers (e.g. kernels/sort.h ->
  # hwy/contrib/sort/vqsort.h). The dataframe library links Highway PRIVATE, so
  # its include dir is not propagated to consumers; bring it in for the tests
  # that include those headers directly.
  if(TARGET hwy)
    target_link_libraries(${target_name} PRIVATE $<BUILD_INTERFACE:hwy>)
  endif()
  if(TARGET hwy_contrib)
    target_link_libraries(${target_name} PRIVATE $<BUILD_INTERFACE:hwy_contrib>)
  endif()
  target_set_warnings(${target_name})
  target_enable_coroutine(${target_name})

  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${bin_exec})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE --coverage -fprofile-arcs
                                                   -ftest-coverage)
    target_link_libraries(${target_name} PRIVATE --coverage)
  endif()

  set(test_working_directory
      "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${target_name}")
  file(MAKE_DIRECTORY "${test_working_directory}")

  add_test(NAME dataframe/${bin_exec} COMMAND ${target_name})
  set_tests_properties(dataframe/${bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${test_working_directory}")
endforeach()
