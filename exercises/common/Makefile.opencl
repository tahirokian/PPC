include ../common/Makefile.common
uname:=$(shell uname -s)
ifeq ($(uname),Darwin)
LDFLAGS+=-framework OpenCL
else
LDFLAGS+=-L/usr/local/cuda/lib64
CXXFLAGS+=-isystem /usr/local/cuda/include
EXTRA_LIBS+=-lOpenCL
endif
