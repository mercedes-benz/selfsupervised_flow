message(STATUS "Running FindTensorflow.cmake")

if(NOT Tensorflow_FOUND)
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import tensorflow; print([el for el in tensorflow.sysconfig.get_compile_flags() if 'ABI' in el][0])" OUTPUT_VARIABLE Tensorflow_ABI_FLAG OUTPUT_STRIP_TRAILING_WHITESPACE)

find_library(
  Tensorflow_LIB
  NAMES libtensorflow_framework.so.2 libtensorflow_framework.so.1 libtensorflow_framework.so
  PATHS ${Tensorflow_LIB_DIR}
  NO_DEFAULT_PATH
  )

# Handle the QUIETLY and REQUIRED arguments and set Tensorflow_FOUND to TRUE if
# all of the listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Tensorflow
    DEFAULT_MSG
    Tensorflow_INCLUDE_DIR
    Tensorflow_LIB
    Tensorflow_ABI_FLAG)

SET(Tensorflow_ABI_FLAG ${Tensorflow_ABI_FLAG} CACHE STRING "ABI FLAG")
SET(Tensorflow_FOUND ${Tensorflow_FOUND} CACHE BOOL "TENSORFLOW_FOUND FLAG")

message(WARNING "MAPPING CMAKE VARIABLES")

SET(Tensorflow_INCLUDE_DIR ${Tensorflow_INCLUDE_DIR} CACHE PATH "path to tensorflow header files")
SET(Tensorflow_LIB ${Tensorflow_LIB} CACHE PATH "libtensorflow.so path")

endif()
