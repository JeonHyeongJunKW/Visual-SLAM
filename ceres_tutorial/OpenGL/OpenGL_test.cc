#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <iostream>


// GL과 GLFW를 사용
// OpenGL 작동 확인 코드 - 빨간 윈도우 출력

// call-back error 정의
static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

// Window input ket call-back event 정의
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

// program main
int main(void)
{
    // use GLFW window
    GLFWwindow* window;
    static const GLfloat red[] = {1.0f, 0.0f, 0.0f, 1.0f};

    glfwSetErrorCallback(error_callback);

    // glfw 초기화 및 실패 종료
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    // 640 480 크기의 Windwo를 생성 - "window name"
    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);

    // window 생성에 실패하면 glfw와 프로그램을 종료
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // glfw와 window를 연결하고 keyboard callback event를 등록
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    // glew 초기화 및 실패 확인
    glewExperimental=GL_TRUE;
    GLenum err=glewInit();
    if(err!=GLEW_OK)
    {
        //Problem: glewInit failed, something is seriously wrong.
        std::cout<<"glewInit failed, aborting."<<std::endl;
    }

    // Window 창을 닫을 때까지 반복
    while (!glfwWindowShouldClose(window))
    {
        glClearBufferfv(GL_COLOR, 0, red);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw에 등록된 Window를 제거하고, glfw를 종료
    glfwDestroyWindow(window);
    glfwTerminate();

    // 프로그램 정상종료
    exit(EXIT_SUCCESS);
}
