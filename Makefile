# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
	DETECTED_OS:= Windows
else
	DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
#
# Use `make CXXFLAGS=-DUSE_ICET_OGL3` to enable OpenGL 3 IceT compositing
#
ifeq ($(DETECTED_OS),Windows)
CXX= g++
else
CXX= mpicxx
endif
override CXXFLAGS+= -Wno-sign-compare -Wno-maybe-uninitialized -std=c++11

# Tests
TEST1= nuclear_station
TEST2= neurons
TEST3= point_cloud

# Set source and output directories
SRCDIR= src
OBJDIR= obj
BINDIR= bin

# Set up include and libray directories
ifeq ($(DETECTED_OS),Windows)
	MPI_INC= $(patsubst %\,%,$(MSMPI_INC))
	MPI_LIB= $(patsubst %\,%,$(MSMPI_LIB64))

	INC= -I"$(MPI_INC)" -I"$(MPI_INC)\x64" -I"$(HOMEPATH)\local\include" -I"$(HOMEPATH)\local\include\freetype2" -I.\include
	LIB= -L"$(MPI_LIB)" -L"$(HOMEPATH)\local\lib" -lmsmpi -lIceTCore -lIceTGL3 -lIceTMPI -lglfw3dll -lglad -lglad_egl -lfreetype
else ifeq ($(DETECTED_OS),Darwin)
	INC= -I$(HOME)/local/include -I/usr/local/include/freetype2 -I./include
	LIB= -L$(HOME)/local/lib -lIceTCore -lIceTGL3 -lIceTMPI -lglfw -lglad -lglad_egl -lfreetype
else
	INC= -I$(HOME)/local/include -I$(HOME)/local/include/freetype2 -I/usr/include/freetype2 -I./include
	LIB= -L$(HOME)/local/lib -lGL -lEGL -lIceTCore -lIceTGL3 -lIceTMPI -lglfw -lglad -lglad_egl -lfreetype -ldl
endif

# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
	mkobjdir:= $(shell if not exist $(OBJDIR) mkdir $(OBJDIR))
	mkobjdir:= $(shell if not exist $(OBJDIR)\$(TEST1) mkdir $(OBJDIR)\$(TEST1))
	mkobjdir:= $(shell if not exist $(OBJDIR)\$(TEST2) mkdir $(OBJDIR)\$(TEST2))
	mkobjdir:= $(shell if not exist $(OBJDIR)\$(TEST3) mkdir $(OBJDIR)\$(TEST3))
	mkbindir:= $(shell if not exist $(BINDIR) mkdir $(BINDIR))

	TEST1_OBJS= $(addprefix $(OBJDIR)\$(TEST1)\, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
	TEST1_EXEC= $(addprefix $(BINDIR)\, $(TEST1).exe)
	TEST2_OBJS= $(addprefix $(OBJDIR)\$(TEST2)\, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
	TEST2_EXEC= $(addprefix $(BINDIR)\, $(TEST2).exe)
	TEST3_OBJS= $(addprefix $(OBJDIR)\$(TEST3)\, main.o glslloader.o imgreader.o)
	TEST3_EXEC= $(addprefix $(BINDIR)\, $(TEST3).exe)
else
	mkdirs:= $(shell mkdir -p $(OBJDIR)/$(TEST1) $(OBJDIR)/$(TEST2) $(OBJDIR)/$(TEST3) $(BINDIR))
	
	TEST1_OBJS= $(addprefix $(OBJDIR)/$(TEST1)/, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
	TEST1_EXEC= $(addprefix $(BINDIR)/, $(TEST1))
	TEST2_OBJS= $(addprefix $(OBJDIR)/$(TEST2)/, main.o glslloader.o directory.o imgreader.o objloader.o textrender.o)
	TEST2_EXEC= $(addprefix $(BINDIR)/, $(TEST2))
	TEST3_OBJS= $(addprefix $(OBJDIR)/$(TEST3)/, main.o glslloader.o imgreader.o)
	TEST3_EXEC= $(addprefix $(BINDIR)/, $(TEST3))
endif

# BUILD EVERYTHING
all: test1 test2 test3

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

#Test 3
test3: $(TEST3_EXEC)

$(TEST3_EXEC): $(TEST3_OBJS)
	$(CXX) -o $@ $^ $(LIB)

ifeq ($(DETECTED_OS),Windows)
$(OBJDIR)\$(TEST3)\\%.o: $(SRCDIR)\$(TEST3)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
else
$(OBJDIR)/$(TEST3)/%.o: $(SRCDIR)/$(TEST3)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
endif

# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	del $(TEST1_OBJS) $(TEST2_OBJS) $(TEST3_OBJS) $(TEST1_EXEC) $(TEST2_EXEC) $(TEST3_EXEC)
else
clean:
	rm -f $(TEST1_OBJS) $(TEST2_OBJS) $(TEST3_OBJS) $(TEST1_EXEC) $(TEST2_EXEC) $(TEST3_EXEC)
endif
