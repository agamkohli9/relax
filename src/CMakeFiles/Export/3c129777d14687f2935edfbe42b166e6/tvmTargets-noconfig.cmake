#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tvm::tvm" for configuration ""
set_property(TARGET tvm::tvm APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(tvm::tvm PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtvm.so"
  IMPORTED_SONAME_NOCONFIG "libtvm.so"
  )

list(APPEND _cmake_import_check_targets tvm::tvm )
list(APPEND _cmake_import_check_files_for_tvm::tvm "${_IMPORT_PREFIX}/lib/libtvm.so" )

# Import target "tvm::tvm_runtime" for configuration ""
set_property(TARGET tvm::tvm_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(tvm::tvm_runtime PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtvm_runtime.so"
  IMPORTED_SONAME_NOCONFIG "libtvm_runtime.so"
  )

list(APPEND _cmake_import_check_targets tvm::tvm_runtime )
list(APPEND _cmake_import_check_files_for_tvm::tvm_runtime "${_IMPORT_PREFIX}/lib/libtvm_runtime.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
