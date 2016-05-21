bin=pngsegment is-test is-benchmark
include ../common/Makefile.cuda

is.o: is.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

pngsegment: pngsegment.o is.o pngio.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ -lpng

is-test: is-test.o is.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

is-benchmark: is-benchmark.o is.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

include Makefile.dep
