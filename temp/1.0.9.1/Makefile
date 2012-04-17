
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= 1.0.9_2

#bug_1.0.1

# Cuda source files (compiled with cudacc)
CUFILES		:= bug.cu
CUDEPS		:= bug_kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= bug.cpp \
			   video_transfer.cpp

# Additional libraries needed by the project
USECUBLAS	:= 1
USECUFFT    := 1
USECURAND	:= 1

################################################################################
# Rules and targets

include ../../common/common.mk
