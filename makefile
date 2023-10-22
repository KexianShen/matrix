# git@github.com:Reference-LAPACK/lapack.git
LAPACK 		= /path/to/lapack

CXX			= g++
CXXFLAG		= -w -std=c++11 -g
INCLUDE		= -I.
SRC			= $(wildcard ./*.cpp)

ifdef blas
CXXFLAG		+= -DBLAS
LDFLAG		+= -L$(LAPACK) -llapacke -llapack -lcblas  -lrefblas  -lm -lgfortran
INCLUDE		+= -I$(LAPACK)/LAPACKE/include -I$(LAPACK)/CBLAS/include
endif

demo: $(SRC)
	@$(CXX) $(CXXFLAG) $(INCLUDE) $(SRC) $(LDFLAG) -o demo

.PHONY: clean
clean:
	@rm demo