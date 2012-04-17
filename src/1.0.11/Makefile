
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= tracking_model
#bug_1.0.1

# Cuda source files (compiled with cudacc)
CUFILES		:= processing.cu
CUDEPS		:= processing_kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= bug.cpp \
			   video_transfer.cpp \
			   imitation_utils.cpp

# Additional libraries needed by the project
USECUBLAS	:= 1
USECUFFT    := 1
USECURAND	:= 1

################################################################################
# Rules and targets

include ../../common/common.mk
