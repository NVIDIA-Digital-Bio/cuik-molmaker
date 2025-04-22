# Simplified Catch.cmake module for integration with CTest
function(catch_discover_tests TARGET)
  get_target_property(target_type ${TARGET} TYPE)
  if(NOT target_type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "catch_discover_tests only works with executables")
  endif()

  # Add the test with the same name as the target
  add_test(NAME ${TARGET} COMMAND ${TARGET})
endfunction() 