set(mathdx_VERSION "22.02.0")
# 6

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mathdx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

list(TRANSFORM mathdx_FIND_COMPONENTS TOLOWER)

# Populate components list when blank or ALL is provided
if(NOT mathdx_FIND_COMPONENTS OR "ALL" IN_LIST mathdx_FIND_COMPONENTS)
    if(NOT mathdx_FIND_COMPONENTS AND mathdx_FIND_REQUIRED)
        set(mathdx_FIND_REQUIRED_ALL TRUE)
    endif()
    set(mathdx_ALL_COMPONENTS TRUE)
    set(mathdx_FIND_COMPONENTS "")

    foreach(comp IN ITEMS cufftdx)
        list(APPEND mathdx_FIND_COMPONENTS ${comp})
        if(mathdx_FIND_REQUIRED_ALL)
            set(mathdx_FIND_REQUIRED_${comp} TRUE)
        endif()
    endforeach()
endif()

# mathdx target - points to /mathdx/include
set_and_check(mathdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/mathdx/22.02/include")
set_and_check(mathdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/mathdx/22.02/include")
include("${CMAKE_CURRENT_LIST_DIR}/mathdx-targets.cmake")
if(NOT mathdx_FIND_QUIETLY)
    message(STATUS "mathDx found: ${mathdx_INCLUDE_DIRS}")
endif()

# cuFFTDx
if("cufftdx" IN_LIST mathdx_FIND_COMPONENTS)
    set(cufftdx_REQUIRED "")
    if(mathdx_FIND_REQUIRED_cufftdx)
        set(cufftdx_REQUIRED "REQUIRED")
    endif()
    set(cufftdx_QUIETLY "")
    if(mathdx_FIND_QUIETLY)
        set(cufftdx_QUIETLY "QUIET")
    endif()
    find_package(cufftdx
        ${cufftdx_REQUIRED}
        ${cufftdx_QUIETLY}
        CONFIG
        PATHS "${CMAKE_CURRENT_LIST_DIR}/../../../include/cufftdx/lib/cmake/cufftdx/"
        NO_DEFAULT_PATH
    )
    if(cufftdx_FOUND)
        set(mathdx_cufftdx_FOUND TRUE)
        add_library(mathdx::cufftdx ALIAS cufftdx::cufftdx)
        set(cufftdx_LIBRARIES mathdx::cufftdx)
        if(NOT mathdx_FIND_QUIETLY)
            message(STATUS "mathDx: cuFFTDx found: ${cufftdx_INCLUDE_DIRS}")
        endif()
    else()
        set(mathdx_cufftdx_FOUND FALSE)
    endif()
endif()

# Check all components
check_required_components(mathdx)
