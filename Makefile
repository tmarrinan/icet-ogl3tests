# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
    DETECTED_OS:= Windows
else
    DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
ifeq ($(DETECTED_OS),Windows)
CXX= g++
else
CXX= mpicxx
endif
CXXFLAGS+= -std=c++11

# Tests
TEST1= nuclear_station
TEST2= point_cloud

# Set source and output directories
SRCDIR= src
OBJDIR= obj
BINDIR= bin

# Set up include and libray directories
ifeq ($(DETECTED_OS),Windows)
    MPI_INC= $(patsubst %\,%,$(MSMPI_INC))
    MPI_LIB= $(patsubst %\,%,$(MSMPI_LIB64))

    INC= -I"$(MPI_INC)" -I"$(MPI_INC)\x64" -I"$(HOMEPATH)\local\include" -I"$(HOMEPATH)\local\include\freetype2" -I.\include
    LIB= -L"$(MPI_LIB)" -L"$(HOMEPATH)\local\lib" -lmsmpi -lIceTCore -lIceTGL3 -lIceTMPI -lglfw3dll -lglad -lfreetype
else ifeq ($(DETECTED_OS),Darwin)
    INC= -I$(HOME)/local/include -I/usr/local/include/freetype2 -I./include
    LIB= -L$(HOME)/local/lib -lIceTCore -lIceTGL3 -lIceTMPI -lglfw -lglad -lfreetype
else
    INC= -I$(HOME)/local/include -I/usr/local/include/freetype2 -I./include
    LIB= -L$(HOME)/local/lib -lGL -lIceTCore -lIceTGL3 -lIceTMPI -lglfw -lglad -lfreetype
endif

# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
    mkobjdir:= $(shell if not exist $(OBJDIR) mkdir $(OBJDIR))
    mkobjdir:= $(shell if not exist $(OBJDIR)\$(TEST1) mkdir $(OBJDIR)\$(TEST1))
    mkobjdir:= $(shell if not exist $(OBJDIR)\$(TEST2) mkdir $(OBJDIR)\$(TEST2))
    mkbindir:= $(shell if not exist $(BINDIR) mkdir $(BINDIR))

    TEST1_OBJS= $(addprefix $(OBJDIR)\$(TEST1)\, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
    TEST1_EXEC= $(addprefix $(BINDIR)\, $(TEST1).exe)
    TEST2_OBJS= $(addprefix $(OBJDIR)\$(TEST2)\, main.o glslloader.o)
    TEST2_EXEC= $(addprefix $(BINDIR)\, $(TEST2).exe)
else
    mkdirs:= $(shell mkdir -p $(OBJDIR)/$(TEST1) $(OBJDIR)/$(TEST2) $(BINDIR))
    
    TEST1_OBJS= $(addprefix $(OBJDIR)/$(TEST1)/, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
    TEST1_EXEC= $(addprefix $(BINDIR)/, $(TEST1))
    TEST2_OBJS= $(addprefix $(OBJDIR)/$(TEST2)/, main.o glslloader.o)
    TEST2_EXEC= $(addprefix $(BINDIR)/, $(TEST2))
endif

# BUILD EVERYTHING
all: test1 test2

# Test 1
test1: $(TEST1_EXEC)

$(TEST1_EXEC): $(TEST1_OBJS)
	$(CXX) -o $@ $^ $(LIB)

ifeq ($(DETECTED_OS),Windows)
$(OBJDIR)\$(TEST1)\\%.o: $(SRCDIR)\$(TEST1)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
else
$(OBJDIR)/$(TEST1)/%.o: $(SRCDIR)/$(TEST1)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
endif

#Test 2
test2: $(TEST2_EXEC)

$(TEST2_EXEC): $(TEST2_OBJS)
	$(CXX) -o $@ $^ $(LIB)

ifeq ($(DETECTED_OS),Windows)
$(OBJDIR)\$(TEST2)\\%.o: $(SRCDIR)\$(TEST2)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
else
$(OBJDIR)/$(TEST2)/%.o: $(SRCDIR)/$(TEST2)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
endif

# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	del $(TEST1_OBJS) $(TEST2_OBJS) $(TEST1_EXEC) $(TEST2_EXEC)
else
clean:
	rm -f $(TEST1_OBJS) $(TEST2_OBJS) $(TEST1_EXEC) $(TEST2_EXEC)
endif
