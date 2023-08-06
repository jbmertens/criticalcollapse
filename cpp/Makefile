# Generic Makefile

CC = g++

SRC = $(wildcard *.cc)
OBJ = $(patsubst %.cc, %.o, $(SRC))

EXEC = ms

CC_OPTS = -fopenmp
CC_LINKS = -lm -fopenmp -I/usr/include/

# enable debug mode
ifeq ($(shared), 1)
	CC_OPTS += -fPIC -shared
	CC_LINKS += -shared
	EXEC = ms.so
endif

# enable debug mode
ifeq ($(debug), 1)
	CC_OPTIMIZE = -O0 -g -Wall -std=c++17
else
	CC_OPTIMIZE = -O2 -std=c++17
endif

# fast math
ifeq ($(fast), 1)
	CC_OPTIMIZE += -ffast-math -flto -fno-finite-math-only
endif

# profiling
ifeq ($(prof), 1)
	CC_OPTS += -pg
endif

# Require all object files and then link
all: $(OBJ)
	$(CC) $(OBJ) -o $(EXEC) $(CC_OPTS) $(CC_OPTIMIZE) $(CC_LINKS)

# Just compile every .cc file
%.o: %.cc
	$(CC) -c $< $(CC_OPTIMIZE) $(CC_OPTS)

clean:
	rm -f $(EXEC) $(OBJ)
