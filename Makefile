CC = nvcc
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37
LIBS = 

TARGETS = gpu_cnn

all:	$(TARGETS)

gpu_cnn: gpu_cnn.o 
	$(CC) -o $@ $(NVCCLIBS) gpu_cnn.o

gpu_cnn.o:	gpu_cnn.cu cnn.h
	$(CC) -c $(NVCCFLAGS) gpu_cnn.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt *.log

