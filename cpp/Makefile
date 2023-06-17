# Generic Makefile

CC = g++

SRC = $(wildcard *.cc)
OBJ = $(patsubst %.cc, %.o, $(SRC))

EXEC = ms

CC_OPTS = -fopenmp
CC_LINKS = -lm -fopenmp


# enable debug mode
ifeq ($(debug), 1)
	CC_OPTIMIZE = -O0 -g -Wall -std=c++17
else
	CC_OPTIMIZE = -O3 -std=c++17
endif

# fast math
ifeq ($(fast), 1)
	CC_OPTIMIZE += -ffast-math -flto
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
