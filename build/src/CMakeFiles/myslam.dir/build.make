# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yhb/slambook2/ch13

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yhb/slambook2/ch13/build

# Include any dependencies generated for this target.
include src/CMakeFiles/myslam.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/myslam.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/myslam.dir/flags.make

src/CMakeFiles/myslam.dir/frame.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/frame.cpp.o: ../src/frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/myslam.dir/frame.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/frame.cpp.o -c /home/yhb/slambook2/ch13/src/frame.cpp

src/CMakeFiles/myslam.dir/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/frame.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/frame.cpp > CMakeFiles/myslam.dir/frame.cpp.i

src/CMakeFiles/myslam.dir/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/frame.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/frame.cpp -o CMakeFiles/myslam.dir/frame.cpp.s

src/CMakeFiles/myslam.dir/frame.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/frame.cpp.o.requires

src/CMakeFiles/myslam.dir/frame.cpp.o.provides: src/CMakeFiles/myslam.dir/frame.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/frame.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/frame.cpp.o.provides

src/CMakeFiles/myslam.dir/frame.cpp.o.provides.build: src/CMakeFiles/myslam.dir/frame.cpp.o


src/CMakeFiles/myslam.dir/mappoint.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/mappoint.cpp.o: ../src/mappoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/myslam.dir/mappoint.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/mappoint.cpp.o -c /home/yhb/slambook2/ch13/src/mappoint.cpp

src/CMakeFiles/myslam.dir/mappoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/mappoint.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/mappoint.cpp > CMakeFiles/myslam.dir/mappoint.cpp.i

src/CMakeFiles/myslam.dir/mappoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/mappoint.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/mappoint.cpp -o CMakeFiles/myslam.dir/mappoint.cpp.s

src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires

src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides: src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides

src/CMakeFiles/myslam.dir/mappoint.cpp.o.provides.build: src/CMakeFiles/myslam.dir/mappoint.cpp.o


src/CMakeFiles/myslam.dir/map.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/map.cpp.o: ../src/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/myslam.dir/map.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/map.cpp.o -c /home/yhb/slambook2/ch13/src/map.cpp

src/CMakeFiles/myslam.dir/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/map.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/map.cpp > CMakeFiles/myslam.dir/map.cpp.i

src/CMakeFiles/myslam.dir/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/map.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/map.cpp -o CMakeFiles/myslam.dir/map.cpp.s

src/CMakeFiles/myslam.dir/map.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/map.cpp.o.requires

src/CMakeFiles/myslam.dir/map.cpp.o.provides: src/CMakeFiles/myslam.dir/map.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/map.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/map.cpp.o.provides

src/CMakeFiles/myslam.dir/map.cpp.o.provides.build: src/CMakeFiles/myslam.dir/map.cpp.o


src/CMakeFiles/myslam.dir/camera.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/myslam.dir/camera.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/camera.cpp.o -c /home/yhb/slambook2/ch13/src/camera.cpp

src/CMakeFiles/myslam.dir/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/camera.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/camera.cpp > CMakeFiles/myslam.dir/camera.cpp.i

src/CMakeFiles/myslam.dir/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/camera.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/camera.cpp -o CMakeFiles/myslam.dir/camera.cpp.s

src/CMakeFiles/myslam.dir/camera.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/camera.cpp.o.requires

src/CMakeFiles/myslam.dir/camera.cpp.o.provides: src/CMakeFiles/myslam.dir/camera.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/camera.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/camera.cpp.o.provides

src/CMakeFiles/myslam.dir/camera.cpp.o.provides.build: src/CMakeFiles/myslam.dir/camera.cpp.o


src/CMakeFiles/myslam.dir/config.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/myslam.dir/config.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/config.cpp.o -c /home/yhb/slambook2/ch13/src/config.cpp

src/CMakeFiles/myslam.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/config.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/config.cpp > CMakeFiles/myslam.dir/config.cpp.i

src/CMakeFiles/myslam.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/config.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/config.cpp -o CMakeFiles/myslam.dir/config.cpp.s

src/CMakeFiles/myslam.dir/config.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/config.cpp.o.requires

src/CMakeFiles/myslam.dir/config.cpp.o.provides: src/CMakeFiles/myslam.dir/config.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/config.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/config.cpp.o.provides

src/CMakeFiles/myslam.dir/config.cpp.o.provides.build: src/CMakeFiles/myslam.dir/config.cpp.o


src/CMakeFiles/myslam.dir/feature.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/feature.cpp.o: ../src/feature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/myslam.dir/feature.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/feature.cpp.o -c /home/yhb/slambook2/ch13/src/feature.cpp

src/CMakeFiles/myslam.dir/feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/feature.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/feature.cpp > CMakeFiles/myslam.dir/feature.cpp.i

src/CMakeFiles/myslam.dir/feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/feature.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/feature.cpp -o CMakeFiles/myslam.dir/feature.cpp.s

src/CMakeFiles/myslam.dir/feature.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/feature.cpp.o.requires

src/CMakeFiles/myslam.dir/feature.cpp.o.provides: src/CMakeFiles/myslam.dir/feature.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/feature.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/feature.cpp.o.provides

src/CMakeFiles/myslam.dir/feature.cpp.o.provides.build: src/CMakeFiles/myslam.dir/feature.cpp.o


src/CMakeFiles/myslam.dir/frontend.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/frontend.cpp.o: ../src/frontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/myslam.dir/frontend.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/frontend.cpp.o -c /home/yhb/slambook2/ch13/src/frontend.cpp

src/CMakeFiles/myslam.dir/frontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/frontend.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/frontend.cpp > CMakeFiles/myslam.dir/frontend.cpp.i

src/CMakeFiles/myslam.dir/frontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/frontend.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/frontend.cpp -o CMakeFiles/myslam.dir/frontend.cpp.s

src/CMakeFiles/myslam.dir/frontend.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/frontend.cpp.o.requires

src/CMakeFiles/myslam.dir/frontend.cpp.o.provides: src/CMakeFiles/myslam.dir/frontend.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/frontend.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/frontend.cpp.o.provides

src/CMakeFiles/myslam.dir/frontend.cpp.o.provides.build: src/CMakeFiles/myslam.dir/frontend.cpp.o


src/CMakeFiles/myslam.dir/backend.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/backend.cpp.o: ../src/backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/myslam.dir/backend.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/backend.cpp.o -c /home/yhb/slambook2/ch13/src/backend.cpp

src/CMakeFiles/myslam.dir/backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/backend.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/backend.cpp > CMakeFiles/myslam.dir/backend.cpp.i

src/CMakeFiles/myslam.dir/backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/backend.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/backend.cpp -o CMakeFiles/myslam.dir/backend.cpp.s

src/CMakeFiles/myslam.dir/backend.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/backend.cpp.o.requires

src/CMakeFiles/myslam.dir/backend.cpp.o.provides: src/CMakeFiles/myslam.dir/backend.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/backend.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/backend.cpp.o.provides

src/CMakeFiles/myslam.dir/backend.cpp.o.provides.build: src/CMakeFiles/myslam.dir/backend.cpp.o


src/CMakeFiles/myslam.dir/viewer.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/viewer.cpp.o: ../src/viewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/myslam.dir/viewer.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/viewer.cpp.o -c /home/yhb/slambook2/ch13/src/viewer.cpp

src/CMakeFiles/myslam.dir/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/viewer.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/viewer.cpp > CMakeFiles/myslam.dir/viewer.cpp.i

src/CMakeFiles/myslam.dir/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/viewer.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/viewer.cpp -o CMakeFiles/myslam.dir/viewer.cpp.s

src/CMakeFiles/myslam.dir/viewer.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/viewer.cpp.o.requires

src/CMakeFiles/myslam.dir/viewer.cpp.o.provides: src/CMakeFiles/myslam.dir/viewer.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/viewer.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/viewer.cpp.o.provides

src/CMakeFiles/myslam.dir/viewer.cpp.o.provides.build: src/CMakeFiles/myslam.dir/viewer.cpp.o


src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: ../src/visual_odometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/myslam.dir/visual_odometry.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/visual_odometry.cpp.o -c /home/yhb/slambook2/ch13/src/visual_odometry.cpp

src/CMakeFiles/myslam.dir/visual_odometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/visual_odometry.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/visual_odometry.cpp > CMakeFiles/myslam.dir/visual_odometry.cpp.i

src/CMakeFiles/myslam.dir/visual_odometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/visual_odometry.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/visual_odometry.cpp -o CMakeFiles/myslam.dir/visual_odometry.cpp.s

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.provides.build: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o


src/CMakeFiles/myslam.dir/dataset.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/myslam.dir/dataset.cpp.o"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/dataset.cpp.o -c /home/yhb/slambook2/ch13/src/dataset.cpp

src/CMakeFiles/myslam.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/dataset.cpp.i"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yhb/slambook2/ch13/src/dataset.cpp > CMakeFiles/myslam.dir/dataset.cpp.i

src/CMakeFiles/myslam.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/dataset.cpp.s"
	cd /home/yhb/slambook2/ch13/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yhb/slambook2/ch13/src/dataset.cpp -o CMakeFiles/myslam.dir/dataset.cpp.s

src/CMakeFiles/myslam.dir/dataset.cpp.o.requires:

.PHONY : src/CMakeFiles/myslam.dir/dataset.cpp.o.requires

src/CMakeFiles/myslam.dir/dataset.cpp.o.provides: src/CMakeFiles/myslam.dir/dataset.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/myslam.dir/build.make src/CMakeFiles/myslam.dir/dataset.cpp.o.provides.build
.PHONY : src/CMakeFiles/myslam.dir/dataset.cpp.o.provides

src/CMakeFiles/myslam.dir/dataset.cpp.o.provides.build: src/CMakeFiles/myslam.dir/dataset.cpp.o


# Object files for target myslam
myslam_OBJECTS = \
"CMakeFiles/myslam.dir/frame.cpp.o" \
"CMakeFiles/myslam.dir/mappoint.cpp.o" \
"CMakeFiles/myslam.dir/map.cpp.o" \
"CMakeFiles/myslam.dir/camera.cpp.o" \
"CMakeFiles/myslam.dir/config.cpp.o" \
"CMakeFiles/myslam.dir/feature.cpp.o" \
"CMakeFiles/myslam.dir/frontend.cpp.o" \
"CMakeFiles/myslam.dir/backend.cpp.o" \
"CMakeFiles/myslam.dir/viewer.cpp.o" \
"CMakeFiles/myslam.dir/visual_odometry.cpp.o" \
"CMakeFiles/myslam.dir/dataset.cpp.o"

# External object files for target myslam
myslam_EXTERNAL_OBJECTS =

../lib/libmyslam.so: src/CMakeFiles/myslam.dir/frame.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/mappoint.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/map.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/camera.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/config.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/feature.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/frontend.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/backend.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/viewer.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/dataset.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/build.make
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
../lib/libmyslam.so: /usr/local/lib/libpangolin.so
../lib/libmyslam.so: /usr/local/lib/libgtest.a
../lib/libmyslam.so: /usr/local/lib/libgtest_main.a
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libglog.so
../lib/libmyslam.so: /usr/local/lib/libgflags.a
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGL.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libSM.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libICE.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libX11.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libXext.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libdc1394.so
../lib/libmyslam.so: /opt/ros/kinetic/lib/librealsense.so
../lib/libmyslam.so: /usr/lib/libOpenNI.so
../lib/libmyslam.so: /usr/lib/libOpenNI2.so
../lib/libmyslam.so: /opt/ros/kinetic/lib/x86_64-linux-gnu/libuvc.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libpng.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libz.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libtiff.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/liblz4.so
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yhb/slambook2/ch13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX shared library ../../lib/libmyslam.so"
	cd /home/yhb/slambook2/ch13/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myslam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/myslam.dir/build: ../lib/libmyslam.so

.PHONY : src/CMakeFiles/myslam.dir/build

src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/frame.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/mappoint.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/map.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/camera.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/config.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/feature.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/frontend.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/backend.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/viewer.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o.requires
src/CMakeFiles/myslam.dir/requires: src/CMakeFiles/myslam.dir/dataset.cpp.o.requires

.PHONY : src/CMakeFiles/myslam.dir/requires

src/CMakeFiles/myslam.dir/clean:
	cd /home/yhb/slambook2/ch13/build/src && $(CMAKE_COMMAND) -P CMakeFiles/myslam.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/myslam.dir/clean

src/CMakeFiles/myslam.dir/depend:
	cd /home/yhb/slambook2/ch13/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yhb/slambook2/ch13 /home/yhb/slambook2/ch13/src /home/yhb/slambook2/ch13/build /home/yhb/slambook2/ch13/build/src /home/yhb/slambook2/ch13/build/src/CMakeFiles/myslam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/myslam.dir/depend

