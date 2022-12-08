# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/agamserver/eecs583/relax/cmake/libs/../../3rdparty/libbacktrace"
  "/home/agamserver/eecs583/relax/src/libbacktrace"
  "/home/agamserver/eecs583/relax/src/libbacktrace"
  "/home/agamserver/eecs583/relax/src/libbacktrace/tmp"
  "/home/agamserver/eecs583/relax/src/libbacktrace/src/project_libbacktrace-stamp"
  "/home/agamserver/eecs583/relax/src/libbacktrace/src"
  "/home/agamserver/eecs583/relax/src/libbacktrace/src/project_libbacktrace-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/agamserver/eecs583/relax/src/libbacktrace/src/project_libbacktrace-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/agamserver/eecs583/relax/src/libbacktrace/src/project_libbacktrace-stamp${cfgdir}") # cfgdir has leading slash
endif()
