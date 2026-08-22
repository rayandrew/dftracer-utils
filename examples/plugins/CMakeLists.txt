# Example plugins built as loadable modules against the public plugin ABI only
# (the include/ headers); they deliberately link none of the internal libraries,
# so a broken build here is real ABI rot in abi.h / plugin.h.

add_library(event_counter_plugin MODULE event_counter.c)
set_target_properties(
  event_counter_plugin
  PROPERTIES OUTPUT_NAME event_counter
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(event_counter_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(duration_histogram_plugin MODULE duration_histogram.cpp)
set_target_properties(
  duration_histogram_plugin
  PROPERTIES OUTPUT_NAME duration_histogram
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)
target_include_directories(duration_histogram_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_enable_coroutine(duration_histogram_plugin)

add_library(query_filter_plugin MODULE query_filter.cpp)
set_target_properties(
  query_filter_plugin
  PROPERTIES OUTPUT_NAME query_filter
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)
target_include_directories(query_filter_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_enable_coroutine(query_filter_plugin)

add_library(tag_provider_plugin MODULE tag_provider.c)
set_target_properties(
  tag_provider_plugin
  PROPERTIES OUTPUT_NAME tag_provider
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(tag_provider_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(tag_consumer_plugin MODULE tag_consumer.c)
set_target_properties(
  tag_consumer_plugin
  PROPERTIES OUTPUT_NAME tag_consumer
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(tag_consumer_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(perbatch_producer_plugin MODULE perbatch_producer.c)
set_target_properties(
  perbatch_producer_plugin
  PROPERTIES OUTPUT_NAME perbatch_producer
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(perbatch_producer_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(perbatch_consumer_plugin MODULE perbatch_consumer.c)
set_target_properties(
  perbatch_consumer_plugin
  PROPERTIES OUTPUT_NAME perbatch_consumer
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(perbatch_consumer_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(dur_stats_producer_plugin MODULE dur_stats_producer.c)
set_target_properties(
  dur_stats_producer_plugin
  PROPERTIES OUTPUT_NAME dur_stats_producer
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(dur_stats_producer_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(dur_stats_consumer_plugin MODULE dur_stats_consumer.c)
set_target_properties(
  dur_stats_consumer_plugin
  PROPERTIES OUTPUT_NAME dur_stats_consumer
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(dur_stats_consumer_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_counts_plugin MODULE process_counts.c)
set_target_properties(
  process_counts_plugin
  PROPERTIES OUTPUT_NAME process_counts
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_counts_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_file_edges_plugin MODULE process_file_edges.c)
set_target_properties(
  process_file_edges_plugin
  PROPERTIES OUTPUT_NAME process_file_edges
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_file_edges_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_file_edges_wide_plugin MODULE process_file_edges_wide.c)
set_target_properties(
  process_file_edges_wide_plugin
  PROPERTIES OUTPUT_NAME process_file_edges_wide
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_file_edges_wide_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(name_edges_plugin MODULE name_edges.c)
set_target_properties(
  name_edges_plugin
  PROPERTIES OUTPUT_NAME name_edges
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(name_edges_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_file_set_plugin MODULE process_file_set.c)
set_target_properties(
  process_file_set_plugin
  PROPERTIES OUTPUT_NAME process_file_set
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_file_set_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_event_seq_plugin MODULE process_event_seq.c)
set_target_properties(
  process_event_seq_plugin
  PROPERTIES OUTPUT_NAME process_event_seq
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_event_seq_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_pid_durs_plugin MODULE process_pid_durs.c)
set_target_properties(
  process_pid_durs_plugin
  PROPERTIES OUTPUT_NAME process_pid_durs
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_pid_durs_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_file_nested_plugin MODULE process_file_nested.c)
set_target_properties(
  process_file_nested_plugin
  PROPERTIES OUTPUT_NAME process_file_nested
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_file_nested_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_pid_argstats_plugin MODULE process_pid_argstats.c)
set_target_properties(
  process_pid_argstats_plugin
  PROPERTIES OUTPUT_NAME process_pid_argstats
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_pid_argstats_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_bytes_key_plugin MODULE process_bytes_key.c)
set_target_properties(
  process_bytes_key_plugin
  PROPERTIES OUTPUT_NAME process_bytes_key
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_bytes_key_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_topk_plugin MODULE process_topk.c)
set_target_properties(
  process_topk_plugin
  PROPERTIES OUTPUT_NAME process_topk
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_topk_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_approx_topk_plugin MODULE process_approx_topk.c)
set_target_properties(
  process_approx_topk_plugin
  PROPERTIES OUTPUT_NAME process_approx_topk
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_approx_topk_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_sample_plugin MODULE process_sample.c)
set_target_properties(
  process_sample_plugin
  PROPERTIES OUTPUT_NAME process_sample
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_sample_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_join_plugin MODULE process_join.c)
set_target_properties(
  process_join_plugin
  PROPERTIES OUTPUT_NAME process_join
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_join_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(process_regr_stats_plugin MODULE process_regr_stats.c)
set_target_properties(
  process_regr_stats_plugin
  PROPERTIES OUTPUT_NAME process_regr_stats
             PREFIX ""
             POSITION_INDEPENDENT_CODE ON
             C_STANDARD 99
             C_STANDARD_REQUIRED ON)
target_include_directories(process_regr_stats_plugin
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

install(TARGETS event_counter_plugin duration_histogram_plugin
                query_filter_plugin
                tag_provider_plugin tag_consumer_plugin
                perbatch_producer_plugin perbatch_consumer_plugin
                dur_stats_producer_plugin dur_stats_consumer_plugin
                process_counts_plugin process_file_edges_plugin
                process_file_edges_wide_plugin name_edges_plugin
                process_file_set_plugin process_event_seq_plugin
                process_pid_durs_plugin process_file_nested_plugin
                process_pid_argstats_plugin process_bytes_key_plugin
                process_topk_plugin process_approx_topk_plugin
                process_sample_plugin process_join_plugin
                process_regr_stats_plugin
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}/examples/plugins)
