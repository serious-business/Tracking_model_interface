###############################################################################
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
###############################################################################
#
# GPU Computing SDK (CUDA C)
#
###############################################################################

ifeq ($(emu), 1)
  PROJECTS := $(shell find src -name Makefile | xargs grep -L 'USEDRVAPI' | xargs grep -L 'USENEWINTEROP' )
else
  PROJECTS := $(shell find src -name Makefile)
endif

%.ph_build : lib/libcutil.so lib/libparamgl.so lib/librendercheckgl.so shared/libshrutil.so
	$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

lib/libcutil.so:
	@$(MAKE) -C common

lib/libparamgl.so:
	@$(MAKE) -C common -f Makefile_paramgl

lib/librendercheckgl.so:
	@$(MAKE) -C common -f Makefile_rendercheckgl

shared/libshrutil.so:
	@$(MAKE) -C ../shared/ 

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	@$(MAKE) -C common clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	@$(MAKE) -C common clobber
