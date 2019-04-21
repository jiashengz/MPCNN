#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#

COMPILER = $(shell cc --version)
IS_ICC = $(findstring icc, $(COMPILER))

CXX = c++
MPCXX = mpic++
OPENMP = -fopenmp #Note: this is the flag for GNU compilers. Change this to -openmp for Intel compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3 -std=c++11
LIBS = -lm

ifneq (,$(IS_ICC))
	# patch some flags for the intel compiler
	CXX = CC
	MPCXX = CC
	OPENMP = -openmp
	LIBS = 
endif

TARGETS = test

all:	$(TARGETS)

test: test.o 
	$(CXX) -o $@ $(LIBS) test.o
openmp: test_openmp.o 
	$(CXX) -o $@ $(LIBS) $(OPENMP) test_openmp.o

test.o: test.cpp
	$(CXX) -c $(CFLAGS) test.cpp
test_openmp.o: test_openmp.cpp
	$(CXX) -c $(OPENMP) $(CFLAGS) test_openmp.cpp

# serial: serial.o common.o
# 	$(CXX) -o $@ $(LIBS) serial.o common.o
# autograder: autograder.o common.o
# 	$(CXX) -o $@ $(LIBS) autograder.o common.o
# openmp: openmp.o common.o
# 	$(CXX) -o $@ $(LIBS) $(OPENMP) openmp.o common.o
# mpi: mpi.o common.o
# 	$(MPCXX) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o

# autograder.o: autograder.cpp common.h
# 	$(CXX) -c $(CFLAGS) autograder.cpp
# openmp.o: openmp.cpp common.h
# 	$(CXX) -c $(OPENMP) $(CFLAGS) openmp.cpp
# serial.o: serial.cpp common.h
# 	$(CXX) -c $(CFLAGS) serial.cpp
# mpi.o: mpi.cpp common.h
# 	$(MPCXX) -c $(CFLAGS) mpi.cpp
# common.o: common.cpp common.h
# 	$(CXX) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
