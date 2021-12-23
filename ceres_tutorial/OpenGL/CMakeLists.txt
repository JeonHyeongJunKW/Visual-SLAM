cmake_minimum_required(VERSION 2.8)

project(OpenGL_test) # this is case sensitive 

######################################
# finds OpenGL, GLU and X11
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message("ERROR: OpenGL not found")
endif(NOT OPENGL_FOUND)
set(GL_LIBRARY GL GLU GLEW X11)

add_executable(OpenGL_test OpenGL_test.cc)

# linking "glfw" and "glew", not "glfw3" 
# assumes that glfw was built with BUILD_SHARED_LIBS to ON
target_link_libraries(OpenGL_test glfw ${GL_LIBRARY} m)
