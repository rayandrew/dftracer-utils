# ##############################################################################
# Utilities Tests
# ##############################################################################

# This file contains tests for dftracer utilities organized by category.
# Each utility has its own test file for easy maintenance.

# +++++++++++++++++++++++++++++++++++++++++
# Utilities Test Sources
# +++++++++++++++++++++++++++++++++++++++++

set(UTILITIES_TEST_SOURCES

    # Statistics
    common/statistics/test_log2_histogram.cpp
    common/statistics/test_timestamp_histogram.cpp

    # Distribution fitting
    common/statistics/test_distributions.cpp
    common/statistics/test_mixture.cpp

    # DLIO config generation
    dlio/test_barrier_simulator.cpp

    # Sketches
    common/statistics/test_ddsketch.cpp
    common/statistics/test_distinct_sketch.cpp
    common/statistics/test_field_stat.cpp

    # Indexer
    indexer/test_rocksdb_storage.cpp
    indexer/test_scan_prefix.cpp
    indexer/test_index_database.cpp
    indexer/test_index_builder.cpp
    indexer/test_hash_table_aggregation.cpp
    indexer/test_sst_ingest_spike.cpp

    # Compression
    compression/test_libdeflate_gzip.cpp
    compression/test_gzip_rechunker.cpp

    # Filesystem
    filesystem/test_directory_scanner.cpp
    filesystem/test_pattern_directory_scanner.cpp

    # Hash
    hash/test_hasher_utility.cpp

    # I/O
    fileio/test_file_reader.cpp
    fileio/test_streaming_file_writer.cpp
    fileio/test_indexed_file_reader.cpp
    fileio/test_file_compressor.cpp
    fileio/test_file_decompressor.cpp

    # I/O Parallel
    fileio/parallel/test_layout_sizing.cpp
    fileio/parallel/test_striped_writer.cpp
    fileio/parallel/test_sharded_writer.cpp
    fileio/parallel/test_padded_striped_writer.cpp

    # I/O Lines
    fileio/lines/sources/test_indexed_file_line_iterator.cpp
    fileio/lines/sources/test_indexed_file_bytes_iterator.cpp
    fileio/lines/sources/test_async_indexed_file_line_generator.cpp
    fileio/lines/sources/test_async_streaming_gz_line_generator.cpp
    fileio/lines/sources/test_async_indexed_file_bytes_generator.cpp

    # Reader
    reader/test_trace_reader.cpp
    reader/test_sub_chunk_skip.cpp
    reader/test_member_decode_cache.cpp
    reader/test_member_cache_integration.cpp

    # Text
    text/test_line_filter.cpp

    # Plugins
    plugins/test_plugin_host_services.cpp
    plugins/test_plugin_io_writer.cpp
    plugins/test_plugin_cxx_ergonomics.cpp
    plugins/test_plugin_cxx_wrappers.cpp
    plugins/test_plugin_abi_features.cpp
    plugins/test_plugin_ports.cpp
    plugins/test_plugin_results.cpp
    plugins/test_plugin_agg.cpp
    plugins/test_plugin_ops.cpp
    plugins/test_plugin_planner.cpp
)

# +++++++++++++++++++++++++++++++++++++++++
# Arrow Tests (optional, requires nanoarrow)
# +++++++++++++++++++++++++++++++++++++++++

if(DFTRACER_UTILS_ENABLE_ARROW)
  set(ARROW_TEST_SOURCE common/arrow/test_arrow_column_builder.cpp)
  string(REPLACE ".cpp" "" arrow_bin_exec ${ARROW_TEST_SOURCE})
  string(REPLACE "/" "_" arrow_target "utilities_${arrow_bin_exec}")

  add_executable(${arrow_target} ${ARROW_TEST_SOURCE})
  target_include_directories(${arrow_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${arrow_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${arrow_target} STATIC)
  target_set_warnings(${arrow_target})

  get_filename_component(arrow_bin_dir ${arrow_bin_exec} DIRECTORY)
  get_filename_component(arrow_bin_name ${arrow_bin_exec} NAME)
  if(arrow_bin_dir)
    set_target_properties(${arrow_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${arrow_bin_dir}")
  endif()
  set_target_properties(${arrow_target} PROPERTIES OUTPUT_NAME ${arrow_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${arrow_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${arrow_target} PRIVATE --coverage)
  endif()

  set(arrow_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${arrow_target}")
  file(MAKE_DIRECTORY "${arrow_workdir}")
  add_test(NAME utilities/${arrow_bin_exec} COMMAND ${arrow_target})
  set_tests_properties(utilities/${arrow_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${arrow_workdir}")

  set(JOIN_TEST_SOURCE common/arrow/test_arrow_join.cpp)
  string(REPLACE ".cpp" "" join_bin_exec ${JOIN_TEST_SOURCE})
  string(REPLACE "/" "_" join_target "utilities_${join_bin_exec}")

  add_executable(${join_target} ${JOIN_TEST_SOURCE})
  target_include_directories(${join_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${join_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${join_target} STATIC)
  target_set_warnings(${join_target})

  get_filename_component(join_bin_dir ${join_bin_exec} DIRECTORY)
  get_filename_component(join_bin_name ${join_bin_exec} NAME)
  if(join_bin_dir)
    set_target_properties(${join_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${join_bin_dir}")
  endif()
  set_target_properties(${join_target} PROPERTIES OUTPUT_NAME ${join_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${join_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${join_target} PRIVATE --coverage)
  endif()

  set(join_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${join_target}")
  file(MAKE_DIRECTORY "${join_workdir}")
  add_test(NAME utilities/${join_bin_exec} COMMAND ${join_target})
  set_tests_properties(utilities/${join_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${join_workdir}")

  set(WINDOW_TEST_SOURCE common/arrow/test_arrow_window.cpp)
  string(REPLACE ".cpp" "" window_bin_exec ${WINDOW_TEST_SOURCE})
  string(REPLACE "/" "_" window_target "utilities_${window_bin_exec}")

  add_executable(${window_target} ${WINDOW_TEST_SOURCE})
  target_include_directories(${window_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${window_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${window_target} STATIC)
  target_set_warnings(${window_target})

  get_filename_component(window_bin_dir ${window_bin_exec} DIRECTORY)
  get_filename_component(window_bin_name ${window_bin_exec} NAME)
  if(window_bin_dir)
    set_target_properties(${window_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${window_bin_dir}")
  endif()
  set_target_properties(${window_target} PROPERTIES OUTPUT_NAME ${window_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${window_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${window_target} PRIVATE --coverage)
  endif()

  set(window_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${window_target}")
  file(MAKE_DIRECTORY "${window_workdir}")
  add_test(NAME utilities/${window_bin_exec} COMMAND ${window_target})
  set_tests_properties(utilities/${window_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${window_workdir}")

  set(GAPFILL_TEST_SOURCE common/arrow/test_arrow_gapfill.cpp)
  string(REPLACE ".cpp" "" gapfill_bin_exec ${GAPFILL_TEST_SOURCE})
  string(REPLACE "/" "_" gapfill_target "utilities_${gapfill_bin_exec}")

  add_executable(${gapfill_target} ${GAPFILL_TEST_SOURCE})
  target_include_directories(${gapfill_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${gapfill_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${gapfill_target} STATIC)
  target_set_warnings(${gapfill_target})

  get_filename_component(gapfill_bin_dir ${gapfill_bin_exec} DIRECTORY)
  get_filename_component(gapfill_bin_name ${gapfill_bin_exec} NAME)
  if(gapfill_bin_dir)
    set_target_properties(${gapfill_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${gapfill_bin_dir}")
  endif()
  set_target_properties(${gapfill_target} PROPERTIES OUTPUT_NAME ${gapfill_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${gapfill_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${gapfill_target} PRIVATE --coverage)
  endif()

  set(gapfill_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${gapfill_target}")
  file(MAKE_DIRECTORY "${gapfill_workdir}")
  add_test(NAME utilities/${gapfill_bin_exec} COMMAND ${gapfill_target})
  set_tests_properties(utilities/${gapfill_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${gapfill_workdir}")

  set(ATTRIBUTION_TEST_SOURCE common/arrow/test_arrow_attribution.cpp)
  string(REPLACE ".cpp" "" attribution_bin_exec ${ATTRIBUTION_TEST_SOURCE})
  string(REPLACE "/" "_" attribution_target "utilities_${attribution_bin_exec}")

  add_executable(${attribution_target} ${ATTRIBUTION_TEST_SOURCE})
  target_include_directories(${attribution_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${attribution_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${attribution_target} STATIC)
  target_set_warnings(${attribution_target})

  get_filename_component(attribution_bin_dir ${attribution_bin_exec} DIRECTORY)
  get_filename_component(attribution_bin_name ${attribution_bin_exec} NAME)
  if(attribution_bin_dir)
    set_target_properties(${attribution_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${attribution_bin_dir}")
  endif()
  set_target_properties(${attribution_target} PROPERTIES OUTPUT_NAME ${attribution_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${attribution_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${attribution_target} PRIVATE --coverage)
  endif()

  set(attribution_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${attribution_target}")
  file(MAKE_DIRECTORY "${attribution_workdir}")
  add_test(NAME utilities/${attribution_bin_exec} COMMAND ${attribution_target})
  set_tests_properties(utilities/${attribution_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${attribution_workdir}")
endif()

# +++++++++++++++++++++++++++++++++++++++++
# Arrow IPC Tests (optional, requires nanoarrow IPC)
# +++++++++++++++++++++++++++++++++++++++++

if(DFTRACER_UTILS_ENABLE_ARROW_IPC)
  # IPC Writer tests
  set(ARROW_IPC_TEST_SOURCE common/arrow/test_arrow_ipc_writer.cpp)
  string(REPLACE ".cpp" "" arrow_ipc_bin_exec ${ARROW_IPC_TEST_SOURCE})
  string(REPLACE "/" "_" arrow_ipc_target "utilities_${arrow_ipc_bin_exec}")

  add_executable(${arrow_ipc_target} ${ARROW_IPC_TEST_SOURCE})
  target_include_directories(${arrow_ipc_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${arrow_ipc_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${arrow_ipc_target} STATIC)
  target_set_warnings(${arrow_ipc_target})

  get_filename_component(arrow_ipc_bin_dir ${arrow_ipc_bin_exec} DIRECTORY)
  get_filename_component(arrow_ipc_bin_name ${arrow_ipc_bin_exec} NAME)
  if(arrow_ipc_bin_dir)
    set_target_properties(${arrow_ipc_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${arrow_ipc_bin_dir}")
  endif()
  set_target_properties(${arrow_ipc_target} PROPERTIES OUTPUT_NAME ${arrow_ipc_bin_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${arrow_ipc_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${arrow_ipc_target} PRIVATE --coverage)
  endif()

  set(arrow_ipc_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${arrow_ipc_target}")
  file(MAKE_DIRECTORY "${arrow_ipc_workdir}")
  add_test(NAME utilities/${arrow_ipc_bin_exec} COMMAND ${arrow_ipc_target})
  set_tests_properties(utilities/${arrow_ipc_bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${arrow_ipc_workdir}")

  # IPC Reader tests
  set(ARROW_IPC_READER_SOURCE common/arrow/test_arrow_ipc_reader.cpp)
  string(REPLACE ".cpp" "" arrow_ipc_reader_bin ${ARROW_IPC_READER_SOURCE})
  string(REPLACE "/" "_" arrow_ipc_reader_target "utilities_${arrow_ipc_reader_bin}")

  add_executable(${arrow_ipc_reader_target} ${ARROW_IPC_READER_SOURCE})
  target_include_directories(${arrow_ipc_reader_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${arrow_ipc_reader_target} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  link_nanoarrow(${arrow_ipc_reader_target} STATIC)
  target_set_warnings(${arrow_ipc_reader_target})
  target_enable_coroutine(${arrow_ipc_reader_target})

  get_filename_component(arrow_ipc_reader_dir ${arrow_ipc_reader_bin} DIRECTORY)
  get_filename_component(arrow_ipc_reader_name ${arrow_ipc_reader_bin} NAME)
  if(arrow_ipc_reader_dir)
    set_target_properties(${arrow_ipc_reader_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${arrow_ipc_reader_dir}")
  endif()
  set_target_properties(${arrow_ipc_reader_target} PROPERTIES OUTPUT_NAME ${arrow_ipc_reader_name})

  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${arrow_ipc_reader_target} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${arrow_ipc_reader_target} PRIVATE --coverage)
  endif()

  set(arrow_ipc_reader_workdir "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${arrow_ipc_reader_target}")
  file(MAKE_DIRECTORY "${arrow_ipc_reader_workdir}")
  add_test(NAME utilities/${arrow_ipc_reader_bin} COMMAND ${arrow_ipc_reader_target})
  set_tests_properties(utilities/${arrow_ipc_reader_bin} PROPERTIES
                       WORKING_DIRECTORY "${arrow_ipc_reader_workdir}")
endif()

# +++++++++++++++++++++++++++++++++++++++++
# Build Configuration
# +++++++++++++++++++++++++++++++++++++++++

foreach(test_file ${UTILITIES_TEST_SOURCES})
  # Extract path and filename
  string(REPLACE ".cpp" "" bin_exec ${test_file})

  # Create valid target name (replace / with _)
  string(REPLACE "/" "_" target_name "utilities_${bin_exec}")

  add_executable(${target_name} ${test_file})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${target_name} PRIVATE doctest::doctest dftracer_utils testing_utilities)
  target_set_warnings(${target_name})
  target_enable_coroutine(${target_name})

  # Set output directory to preserve folder structure
  get_filename_component(bin_dir ${bin_exec} DIRECTORY)
  get_filename_component(bin_name ${bin_exec} NAME)

  if(bin_dir)
    set_target_properties(${target_name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${bin_dir}")
  endif()

  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${bin_name})

  # Apply coverage flags if enabled
  if(DFTRACER_UTILS_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE --coverage -fprofile-arcs
                                                   -ftest-coverage)
    target_link_libraries(${target_name} PRIVATE --coverage)
  endif()

  set(test_working_directory
      "${CMAKE_CURRENT_BINARY_DIR}/workdirs/${target_name}")
  file(MAKE_DIRECTORY "${test_working_directory}")

  add_test(NAME utilities/${bin_exec} COMMAND ${target_name})
  set_tests_properties(utilities/${bin_exec} PROPERTIES
                       WORKING_DIRECTORY "${test_working_directory}")
endforeach()

# +++++++++++++++++++++++++++++++++++++++++
# Plugin C-ABI test: per-ABI C consumers (compiled as C) driven by a C++ host
# harness, so the plugin C ABIs are exercised from real C, not just C++.
# +++++++++++++++++++++++++++++++++++++++++

set(cabi_target utilities_plugins_test_plugin_c_abi)
set(cabi_sources plugins/test_plugin_c_abi.cpp
                 plugins/compose_abi_c_consumer.c
                 plugins/query_abi_c_consumer.c
                 plugins/sketch_abi_c_consumer.c
                 plugins/ops_abi_c_consumer.c
                 plugins/trace_abi_c_consumer.c
                 plugins/io_abi_c_consumer.c
                 plugins/writer_abi_c_consumer.c)
if(DFTRACER_UTILS_ENABLE_ARROW)
  list(APPEND cabi_sources plugins/arrow_abi_c_consumer.c)
endif()
add_executable(${cabi_target} ${cabi_sources})
target_include_directories(${cabi_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(${cabi_target} PRIVATE doctest::doctest dftracer_utils
                                             testing_utilities)
target_enable_coroutine(${cabi_target})
if(DFTRACER_UTILS_ENABLE_ARROW)
  target_compile_definitions(${cabi_target} PRIVATE DFTU_TEST_HAS_ARROW)
  link_nanoarrow(${cabi_target} STATIC)
endif()
set_target_properties(${cabi_target} PROPERTIES OUTPUT_NAME test_plugin_c_abi
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins")
add_test(NAME utilities/plugins/test_plugin_c_abi COMMAND ${cabi_target})
