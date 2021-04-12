NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61  -maxrregcount="26"
LDFLAGS  := -lm
EXES     := hw4-2

alls: $(EXES)

clean:
	rm -f $(EXES)

hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -Xcompiler=-fopenmp -o $@ $?
