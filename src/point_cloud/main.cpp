#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <IceT.h>
#include <IceTGL3.h>
#include <IceTMPI.h>
#include "glslloader.h"
#include "imgreader.h"


#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define WINDOW_TITLE "Point Cloud Renderer"


typedef struct GlslProgram {
    GLuint program;
    std::map<std::string,GLint> uniforms;
} GlslProgram;

typedef struct Scene {
    glm::vec3 camera_position;
    glm::vec3 camera_target;
    int num_lights;
    GLfloat *light_positions;
    GLfloat *light_colors;
    int num_points;
    GLuint pointcloud_vertex_array;
    int pointcloud_face_index_count;
    glm::dvec3 pointcloud_center;
} Scene;

typedef struct AppData {
    // MPI info
    int rank;
    int num_proc;
    // OpenGL window
    int window_width;
    int window_height;
    GLFWwindow *window;
    // IceT info
    IceTCommunicator comm;
    IceTContext context;
    IceTImage image;
    // Rendering info
    std::map<std::string, GlslProgram> glsl_program;
    GLuint vertex_position_attrib;
    GLuint vertex_texcoord_attrib;
    GLuint point_center_attrib;
    GLuint point_color_attrib;
    GLuint point_size_attrib;
    GLuint background_texture;
    GLuint composite_texture;
    GLuint plane_vertex_array;
    GLuint framebuffer;           // only used in IceT generic compositing
    GLuint framebuffer_texture;   // only used in IceT generic compositing
    GLuint framebuffer_depth;     // only used in IceT generic compositing
    // Frame counter
    int frame_count;
    // Scene info
    glm::vec4 background_color;
    glm::dmat4 projection_matrix;
    glm::dmat4 view_matrix;
    glm::dmat4 model_matrix;
    glm::mat4 composite_projection_matrix;
    glm::mat4 composite_modelview_matrix;
    glm::mat4 background_modelview_matrix;
    double rotate_y;
    Scene scene;
} AppData;


void parseCommandLineArgs(int argc, char **argv);
void init();
void doFrame();
void renderIceTOGL3(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
                    const IceTInt *readback_viewport, IceTUInt framebuffer_id);
void renderIceTGeneric(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
                       const IceTFloat *background_color, const IceTInt *readback_viewport,
                       IceTImage result);
void render();
void display();
void mat4ToFloatArray(glm::dmat4 mat4, float array[16]);
void loadPointCloudShader();
void loadCompositeShader();
void loadPointCloudData(const char *filename, float bbox[6]);
GLuint createPointCloudVertexArray(GLfloat *point_centers, GLfloat *point_colors, GLfloat *point_sizes);
GLuint createPlaneVertexArray();

AppData app;

int main(int argc, char **argv)
{
    // Initialize MPI
    int rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &(app.rank));
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &(app.num_proc));
    if (rc != 0)
    {
        fprintf(stderr, "Error initializing MPI\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Parse command line parameters (or use defaults)
    parseCommandLineArgs(argc, argv);

    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Error: could not initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    // Create a window and its OpenGL context
    char title[32];
    snprintf(title, 32, "%s (%d)", WINDOW_TITLE, app.rank);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (app.rank == 0)
    {
        app.window = glfwCreateWindow(app.window_width, app.window_height, title, NULL, NULL);
    }
    else
    {
        app.window = glfwCreateWindow(320, 180, title, NULL, NULL);
    }

    // Make window's context current
    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1);

    // Initialize GLAD OpenGL extension handling
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        fprintf(stderr, "Error: could not initialize GLAD\n");
        exit(EXIT_FAILURE);
    }

    // Initialize app
    init();

    // Main render loop
    uint16_t should_close = 0;
    while (!should_close)
    {
        // Render frame
        doFrame();

        // Poll for user events
        glfwPollEvents();

        // check if any window has been closed
        uint16_t close_this = glfwWindowShouldClose(app.window);
        MPI_Allreduce(&close_this, &should_close, 1, MPI_UINT16_T, MPI_SUM, MPI_COMM_WORLD);
    }

    // Clean up
    icetDestroyMPICommunicator(app.comm);
    icetDestroyContext(app.context);
    glfwDestroyWindow(app.window);
    glfwTerminate();
    MPI_Finalize();

    return 0;
}

void parseCommandLineArgs(int argc, char **argv)
{
    // Defaults
    app.window_width = 1280;
    app.window_height = 720;

    // User options
    int i = 1;
    while (i < argc)
    {
        std::string argument = argv[i];
        if ((argument == "--width" || argument == "-w") && i < argc - 1)
        {
            app.window_width = std::stoi(argv[i + 1]);
            i += 2;
        }
        else if ((argument == "--height" || argument == "-h") && i < argc - 1)
        {
            app.window_height = std::stoi(argv[i + 1]);
            i += 2;
        }
        else
        {
            i += 1;
        }
    }
}

void init()
{
#ifdef USE_ICET_OGL3
    printf("[Rank % 3d] Using IceT OGL3 Interface\n", app.rank);
#else
    printf("[Rank % 3d] Using IceT Generic Rendering Interface\n", app.rank);
#endif

    // Initialize IceT
    app.comm = icetCreateMPICommunicator(MPI_COMM_WORLD);
    app.context = icetCreateContext(app.comm);
#ifdef USE_ICET_OGL3
    icetGL3Initialize();
#endif

    // Set IceT window configurations
    icetResetTiles();
    icetAddTile(0, 0, app.window_width, app.window_height, 0);

    // Set IceT compositing strategy
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); // best for a single tile
    //icetStrategy(ICET_STRATEGY_REDUCE); // good all around performance for multiple tiles

    // Set IceT framebuffer settings
    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);

    // Set IceT draw callback (main render function)
#ifdef USE_ICET_OGL3
    icetGL3DrawCallbackTexture(renderIceTOGL3);
#else
    glGenTextures(1, &(app.framebuffer_texture));
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &(app.framebuffer_depth));
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, app.window_width, app.window_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &(app.framebuffer));
    glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, app.framebuffer_texture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, app.framebuffer_depth, 0);
    GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, draw_buffers);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    icetDrawCallback(renderIceTGeneric);
#endif

    // Initialize frame count
    app.frame_count = 0;

    // Initialize OpenGL stuff
    app.background_color = glm::vec4(0.0, 0.0, 0.0, 0.0);
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    glViewport(0, 0, app.window_width, app.window_height);

    // Set GLSL attributes
    app.vertex_position_attrib = 0;
    app.vertex_texcoord_attrib = 1;
    app.point_center_attrib = 2;
    app.point_color_attrib = 3;
    app.point_size_attrib = 4;

    // Scene rotation
    app.rotate_y = 0.0;

    // Create composite texture (for display of final image)
    if (app.rank == 0)
    {
        app.plane_vertex_array = createPlaneVertexArray();

        glGenTextures(1, &(app.composite_texture));
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    
        // Background texture
        int bg_w, bg_h;
        uint8_t *bg_pixels;
        imageFileToRgba("resrc/images/globe_bg2.png", &bg_w, &bg_h, &bg_pixels);
        
        glGenTextures(1, &(app.background_texture));
        glBindTexture(GL_TEXTURE_2D, app.background_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bg_w, bg_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        freeRgba(bg_pixels);
    }

    // Load GLSL shader program
    loadPointCloudShader();
    loadCompositeShader();

    // Load point cloud data
    float bbox[6];
    loadPointCloudData("resrc/data/osm_gps_2012.pcd", bbox);
#ifdef USE_ICET_OGL3
    icetBoundingBoxf(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
#endif
    printf("[Rank % 3d] Point Cloud Bounding-Box: x = [%.2f, %.2f], y = [%.2f, %.2f], z = [%.2f, %.2f]\n",
           app.rank, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
    float x_min, y_min, z_min, x_max, y_max, z_max;
    MPI_Allreduce(&(bbox[0]), &x_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&(bbox[2]), &y_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&(bbox[4]), &z_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&(bbox[1]), &x_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&(bbox[3]), &y_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&(bbox[5]), &z_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    app.scene.pointcloud_center = glm::vec3((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0);

    // Create projection, view, and model matrices
    float clip_z[2] = {0.1, 100.0};
    app.projection_matrix = glm::perspective(glm::radians(60.0), (double)app.window_width / (double)app.window_height, (double)clip_z[0], (double)clip_z[1]);
    app.view_matrix = glm::lookAt(app.scene.camera_position, app.scene.camera_target, glm::vec3(0.0, 1.0, 0.0));
    app.model_matrix = glm::translate(glm::dmat4(1.0), app.scene.pointcloud_center);
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(23.5), glm::dvec3(1.0, 0.0, 0.0));
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(15.0), glm::dvec3(0.0, 0.0, 1.0));
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(app.rotate_y), glm::dvec3(0.0, 1.0, 0.0));
    app.model_matrix = glm::translate(app.model_matrix, -app.scene.pointcloud_center);

    // Create ortho display projection and modelview matrices
    app.composite_projection_matrix = glm::ortho(0.0f, (float)app.window_width, 0.0f, (float)app.window_height, -1.0f, 1.0f);
    app.composite_modelview_matrix = glm::translate(glm::mat4(1.0), glm::vec3((float)app.window_width / 2.0f, (float)app.window_height / 2.0f, -0.5f));
    app.composite_modelview_matrix = glm::scale(app.composite_modelview_matrix, glm::vec3((float)app.window_width, (float)app.window_height, 1.0f));
    app.background_modelview_matrix = glm::translate(glm::mat4(1.0), glm::vec3((float)app.window_width / 2.0f, (float)app.window_height / 2.0f, -0.75f));
    app.background_modelview_matrix = glm::scale(app.background_modelview_matrix, glm::vec3((float)app.window_width, (float)app.window_height, 1.0f));

    float ambient[3] = {0.35, 0.35, 0.35};

    float mat4_proj[16], mat4_model[16], mat4_view[16];
    mat4ToFloatArray(app.projection_matrix, mat4_proj);
    mat4ToFloatArray(app.model_matrix, mat4_model);
    mat4ToFloatArray(app.view_matrix, mat4_view);

    glUseProgram(app.glsl_program["pointcloud"].program);
    glUniformMatrix4fv(app.glsl_program["pointcloud"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.glsl_program["pointcloud"].uniforms["view_matrix"], 1, GL_FALSE, mat4_view);
    glUniformMatrix4fv(app.glsl_program["pointcloud"].uniforms["model_matrix"], 1, GL_FALSE, mat4_model);
    glUniform2fv(app.glsl_program["pointcloud"].uniforms["clip_z"], 1, clip_z);
    glUniform3fv(app.glsl_program["pointcloud"].uniforms["camera_position"], 1, glm::value_ptr(app.scene.camera_position));
    glUniform3fv(app.glsl_program["pointcloud"].uniforms["light_ambient"], 1, ambient);
    glUniform1i(app.glsl_program["pointcloud"].uniforms["num_lights"], app.scene.num_lights);
    glUniform3fv(app.glsl_program["pointcloud"].uniforms["light_position[0]"], app.scene.num_lights, app.scene.light_positions);
    glUniform3fv(app.glsl_program["pointcloud"].uniforms["light_color[0]"], app.scene.num_lights, app.scene.light_colors);
    
    glUseProgram(app.glsl_program["nolight"].program);
    glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["projection_matrix"], 1, GL_FALSE, glm::value_ptr(app.composite_projection_matrix));
    glUseProgram(0);
}

void doFrame()
{
    // Offscreen render and composit
    glm::dmat4 modelview_matrix = app.view_matrix * app.model_matrix;
#ifdef USE_ICET_OGL3
    app.image = icetGL3DrawFrame(glm::value_ptr(app.projection_matrix),
                                 glm::value_ptr(modelview_matrix));
#else
    app.image = icetDrawFrame(glm::value_ptr(app.projection_matrix),
                              glm::value_ptr(app.view_matrix),
                              glm::value_ptr(app.background_color));
#endif

    // Render composited image to fullscreen quad on screen of rank 0
    display();

    // Animate
    app.rotate_y -= 0.25;
    app.model_matrix = glm::translate(glm::dmat4(1.0), app.scene.pointcloud_center);
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(23.5), glm::dvec3(1.0, 0.0, 0.0));
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(15.0), glm::dvec3(0.0, 0.0, 1.0));
    app.model_matrix = glm::rotate(app.model_matrix, glm::radians(app.rotate_y), glm::dvec3(0.0, 1.0, 0.0));
    app.model_matrix = glm::translate(app.model_matrix, -app.scene.pointcloud_center);

    float mat4_model[16];
    mat4ToFloatArray(app.model_matrix, mat4_model);
    glUseProgram(app.glsl_program["pointcloud"].program);
    glUniformMatrix4fv(app.glsl_program["pointcloud"].uniforms["model_matrix"], 1, GL_FALSE, mat4_model);
    glUseProgram(0);
}

void renderIceTOGL3(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
                    const IceTInt *readback_viewport, IceTUInt framebuffer_id)
{
    // Render to IceT framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id);

    // Render
    render();

    // Deselect IceT framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void renderIceTGeneric(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
                       const IceTFloat *background_color, const IceTInt *readback_viewport,
                       IceTImage result)
{
    // Render to app's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);

    // Render
    render();

    // Deselect IceT framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Copy image to IceT buffer
    IceTUByte *pixels = icetImageGetColorub(result);
    IceTFloat *depth = icetImageGetDepthf(result);

    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_depth);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void render()
{
    // Clear frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Select shader program to use
    glUseProgram(app.glsl_program["pointcloud"].program);

    // Render
    glBindVertexArray(app.scene.pointcloud_vertex_array);
    glDrawElementsInstanced(GL_TRIANGLES, app.scene.pointcloud_face_index_count, GL_UNSIGNED_SHORT,
                            0, app.scene.num_points);
    glBindVertexArray(0);

    // Deseclect shader program
    glUseProgram(0);
}

void display()
{
    glClearColor(0.235, 0.235, 0.235, 1.000);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);

    if (app.rank == 0)
    {
        glUseProgram(app.glsl_program["nolight"].program);

        glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["modelview_matrix"], 1, GL_FALSE, glm::value_ptr(app.background_modelview_matrix));
   
        glBindVertexArray(app.plane_vertex_array);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, app.background_texture);
        glUniform1i(app.glsl_program["nolight"].uniforms["image"], 0);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        
        glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["modelview_matrix"], 1, GL_FALSE, glm::value_ptr(app.composite_modelview_matrix));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        IceTUByte *pixels = icetImageGetColorub(app.image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.window_width, app.window_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glUniform1i(app.glsl_program["nolight"].uniforms["image"], 0);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);

        glUseProgram(0);
    }

    // Display frame
    glfwSwapBuffers(app.window);
}

void mat4ToFloatArray(glm::dmat4 mat4, float array[16])
{
    array[0] = mat4[0][0];
    array[1] = mat4[0][1];
    array[2] = mat4[0][2];
    array[3] = mat4[0][3];
    array[4] = mat4[1][0];
    array[5] = mat4[1][1];
    array[6] = mat4[1][2];
    array[7] = mat4[1][3];
    array[8] = mat4[2][0];
    array[9] = mat4[2][1];
    array[10] = mat4[2][2];
    array[11] = mat4[2][3];
    array[12] = mat4[3][0];
    array[13] = mat4[3][1];
    array[14] = mat4[3][2];
    array[15] = mat4[3][3];
}

void loadPointCloudShader()
{
    // Compile GPU program
    GlslProgram p;
    p.program = glsl::createShaderProgram("resrc/shaders/pointcloud_color.vert",
                                          "resrc/shaders/pointcloud_color.frag");

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(p.program, app.vertex_position_attrib, "vertex_position");
    glBindAttribLocation(p.program, app.vertex_texcoord_attrib, "vertex_texcoord");
    glBindAttribLocation(p.program, app.point_center_attrib, "point_center");
    glBindAttribLocation(p.program, app.point_color_attrib, "point_color");
    glBindFragDataLocation(p.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(p.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(p.program, p.uniforms);

    // Store GPU program and uniforms
    app.glsl_program["pointcloud"] = p;
}

void loadCompositeShader()
{
    // Compile GPU program
    GlslProgram p;
    p.program = glsl::createShaderProgram("resrc/shaders/nolight_texture.vert",
                                          "resrc/shaders/nolight_texture.frag");

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(p.program, app.vertex_position_attrib, "vertex_position");
    glBindAttribLocation(p.program, app.vertex_texcoord_attrib, "vertex_texcoord");
    glBindFragDataLocation(p.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(p.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(p.program, p.uniforms);

    // Store GPU program and uniforms
    app.glsl_program["nolight"] = p;
}

void loadPointCloudData(const char *filename, float bbox[6])
{
    const int CAMERA_POSITION = 0;
    const int CAMERA_TARGET = 1;
    const int LIGHT_COUNT = 2;
    const int LIGHTS = 3;
    const int POINT_COUNT = 4;
    const int POINTS = 5;

    std::ifstream scene_file(filename, std::ios::binary);
    std::string line;
    int i;
    int section = CAMERA_POSITION;
    int light_idx = 0;
    uint32_t total_points;
    GLfloat *point_centers, *point_colors, *point_sizes;
    while (section != POINTS)
    {
        std::getline(scene_file, line);
        std::istringstream iss(line);
        uint32_t count;
        float x, y, z, red, green, blue;
        switch (section)
        {
            case CAMERA_POSITION:
                iss >> x >> y >> z;
                app.scene.camera_position = glm::vec3(x, y, z);
                section = CAMERA_TARGET;
                break;
            case CAMERA_TARGET:
                iss >> x >> y >> z;
                app.scene.camera_target = glm::vec3(x, y, z);
                section = LIGHT_COUNT;
                break;
            case LIGHT_COUNT:
                iss >> count;
                app.scene.num_lights = count;
                app.scene.light_positions = new GLfloat[3 * app.scene.num_lights];
                app.scene.light_colors = new GLfloat[3 * app.scene.num_lights];
                section = LIGHTS;
                break;
            case LIGHTS:
                iss >> x >> y >> z >> red >> green >> blue;
                app.scene.light_positions[3 * light_idx] = x;
                app.scene.light_positions[3 * light_idx + 1] = y;
                app.scene.light_positions[3 * light_idx + 2] = z;
                app.scene.light_colors[3 * light_idx] = red;
                app.scene.light_colors[3 * light_idx + 1] = green;
                app.scene.light_colors[3 * light_idx + 2] = blue;
                light_idx++;
                if (light_idx >= app.scene.num_lights)
                {
                    section = POINT_COUNT;
                }
                break;
            case POINT_COUNT:
                iss >> count;
                total_points = count;
                section = POINTS;
                break;
        }
    }

    uint32_t points_per_rank = total_points / app.num_proc;
    uint32_t extra_points = total_points % app.num_proc;
    app.scene.num_points = (app.rank < (app.num_proc - 1)) ? points_per_rank : points_per_rank + extra_points;
    point_centers = new GLfloat[3 * app.scene.num_points];
    point_colors = new GLfloat[3 * app.scene.num_points];
    point_sizes = new GLfloat[app.scene.num_points];
    uint32_t point_idx_start = app.rank * points_per_rank;
    uint32_t point_idx_end = point_idx_start + app.scene.num_points;

    scene_file.seekg(point_idx_start * 3 * sizeof(GLfloat), std::ios_base::cur);
    scene_file.read((char*)point_centers, 3 * app.scene.num_points * sizeof(GLfloat));
    scene_file.seekg((total_points - point_idx_end) * 3 * sizeof(GLfloat), std::ios_base::cur);

    scene_file.seekg(point_idx_start * 3 * sizeof(GLfloat), std::ios_base::cur);
    scene_file.read((char*)point_colors, 3 * app.scene.num_points * sizeof(GLfloat));
    scene_file.seekg((total_points - point_idx_end) * 3 * sizeof(GLfloat), std::ios_base::cur);

    scene_file.seekg(point_idx_start * sizeof(GLfloat), std::ios_base::cur);
    scene_file.read((char*)point_sizes, app.scene.num_points * sizeof(GLfloat));

    scene_file.close();

    bbox[0] =  9.9e12; // x min
    bbox[1] = -9.9e12; // x max
    bbox[2] =  9.9e12; // y min
    bbox[3] = -9.9e12; // y max
    bbox[4] =  9.9e12; // z min
    bbox[5] = -9.9e12; // z max
    for (i = 0; i < app.scene.num_points; i++)
    {
        if (point_centers[3 * i + 0] < bbox[0]) bbox[0] = point_centers[3 * i + 0];
        if (point_centers[3 * i + 0] > bbox[1]) bbox[1] = point_centers[3 * i + 0];
        if (point_centers[3 * i + 1] < bbox[2]) bbox[2] = point_centers[3 * i + 1];
        if (point_centers[3 * i + 1] > bbox[3]) bbox[3] = point_centers[3 * i + 1];
        if (point_centers[3 * i + 2] < bbox[4]) bbox[4] = point_centers[3 * i + 2];
        if (point_centers[3 * i + 2] > bbox[5]) bbox[5] = point_centers[3 * i + 2];
    }

    /*
    // ---------------------------------------------------------
    printf("CAMERA: pos = (%.3f, %.3f, %.3f) target = (%.3f, %.3f, %.3f)\n",
           app.scene.camera_position[0], app.scene.camera_position[1], app.scene.camera_position[2],
           app.scene.camera_target[0], app.scene.camera_target[1], app.scene.camera_target[2]);
    printf("LIGHTS (%d):\n", app.scene.num_lights);
    for (int i = 0; i < app.scene.num_lights; i++)
    {
        printf("  pos = (%.3f, %.3f, %.3f) color = (%.3f, %.3f, %.3f)\n",
               app.scene.light_positions[3 * i], app.scene.light_positions[3 * i + 1], app.scene.light_positions[3 * i + 2],
               app.scene.light_colors[3 * i], app.scene.light_colors[3 * i + 1], app.scene.light_colors[3 * i + 2]);
    }
    printf("POINTS (%d):\n", app.scene.num_points);
    for (int i = 0; i < app.scene.num_points; i++)
    {
        printf("  pos = (%.3f, %.3f, %.3f) color = (%.3f, %.3f, %.3f)\n",
               point_centers[3 * i], point_centers[3 * i + 1], point_centers[3 * i + 2],
               point_colors[3 * i], point_colors[3 * i + 1], point_colors[3 * i + 2]);
    }
    // ---------------------------------------------------------
    */

    app.scene.pointcloud_vertex_array = createPointCloudVertexArray(point_centers, point_colors, point_sizes);
    delete[] point_centers;
    delete[] point_colors;
    delete[] point_sizes;
}

GLuint createPointCloudVertexArray(GLfloat *point_centers, GLfloat *point_colors, GLfloat *point_sizes)
{
    // Create a new Vertex Array Object
    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    // Set newly created Vertex Array Object as the active one we are modifying
    glBindVertexArray(vertex_array);

    // Calculate vertices, texture coordinate, and faces
    int num_verts = 4;
    int num_faces = 2;
    GLfloat vertices[12] = {
        -0.5, -0.5,  0.0,
         0.5, -0.5,  0.0,
         0.5,  0.5,  0.0,
        -0.5,  0.5,  0.0
    };
    GLfloat texcoords[8] = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    GLushort indices[6] = {
        0, 1, 2,
        0, 2, 3
    };

    // Create buffer to store vertex positions (3D points)
    GLuint vertex_position_buffer;
    glGenBuffers(1, &vertex_position_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);
    // Store array of vertex positions in the vertex_position_buffer
    glBufferData(GL_ARRAY_BUFFER, 3 * num_verts * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    // Enable vertex_position_attrib in our GPU programs
    glEnableVertexAttribArray(app.vertex_position_attrib);
    // Attach vertex_position_buffer to the vertex_position_attrib
    // (as 3-component floating point values)
    glVertexAttribPointer(app.vertex_position_attrib, 3, GL_FLOAT, false, 0, 0);

    // Create buffer to store texture coordinates (2D coordinates for mapping images to the surface)
    GLuint vertex_texcoord_buffer;
    glGenBuffers(1, &vertex_texcoord_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ARRAY_BUFFER, vertex_texcoord_buffer);
    // Store array of vertex texture coordinates in the vertex_texcoord_buffer
    glBufferData(GL_ARRAY_BUFFER, 2 * num_verts * sizeof(GLfloat), texcoords, GL_STATIC_DRAW);
    // Enable vertex_texcoord_attrib in our GPU program
    glEnableVertexAttribArray(app.vertex_texcoord_attrib);
    // Attach vertex_texcoord_buffer to the vertex_texcoord_attrib
    // (as 2-component floating point values)
    glVertexAttribPointer(app.vertex_texcoord_attrib, 2, GL_FLOAT, false, 0, 0);

    // Create buffer to store faces of the triangle
    GLuint vertex_index_buffer;
    glGenBuffers(1, &vertex_index_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_index_buffer);
    // Store array of vertex indices in the vertex_index_buffer
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * num_faces * sizeof(GLushort), indices, GL_STATIC_DRAW);

    // Point cloud data
    // Create buffer to store point center positions
    GLuint point_center_buffer;
    glGenBuffers(1, &point_center_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ARRAY_BUFFER, point_center_buffer);
    // Store array of point centers in the point_center_buffer
    glBufferData(GL_ARRAY_BUFFER, 3 * app.scene.num_points * sizeof(GLfloat), point_centers, GL_STATIC_DRAW);
    // Enable point_center_attrib in our GPU program
    glEnableVertexAttribArray(app.point_center_attrib);
    // Attach point_center_buffer to the point_center_attrib
    // (as 3-component floating point values)
    glVertexAttribPointer(app.point_center_attrib, 3, GL_FLOAT, false, 0, 0);
    // advance one vertex attribute per instance
    glVertexAttribDivisor(app.point_center_attrib, 1);

    // Create buffer to store point colors
    GLuint point_color_buffer;
    glGenBuffers(1, &point_color_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ARRAY_BUFFER, point_color_buffer);
    // Store array of point colors in the point_color_buffer
    glBufferData(GL_ARRAY_BUFFER, 3 * app.scene.num_points * sizeof(GLfloat), point_colors, GL_STATIC_DRAW);
    // Enable point_color_attrib in our GPU program
    glEnableVertexAttribArray(app.point_color_attrib);
    // Attach point_color_buffer to the point_color_attrib
    // (as 3-component floating point values)
    glVertexAttribPointer(app.point_color_attrib, 3, GL_FLOAT, false, 0, 0);
    // advance one vertex attribute per instance
    glVertexAttribDivisor(app.point_color_attrib, 1);

    // Create buffer to store point sizes
    GLuint point_size_buffer;
    glGenBuffers(1, &point_size_buffer);
    // Set newly created buffer as the active one we are modifying
    glBindBuffer(GL_ARRAY_BUFFER, point_size_buffer);
    // Store array of point colors in the point_color_buffer
    glBufferData(GL_ARRAY_BUFFER, app.scene.num_points * sizeof(GLfloat), point_sizes, GL_STATIC_DRAW);
    // Enable point_color_attrib in our GPU program
    glEnableVertexAttribArray(app.point_size_attrib);
    // Attach point_color_buffer to the point_color_attrib
    // (as 1-component floating point values)
    glVertexAttribPointer(app.point_size_attrib, 1, GL_FLOAT, false, 0, 0);
    // advance one vertex attribute per instance
    glVertexAttribDivisor(app.point_size_attrib, 1);

    // No longer modifying our Vertex Array Object, so deselect
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Store the number of vertices used for entire model (number of faces * 3)
    app.scene.pointcloud_face_index_count = 3 * num_faces;

    // Return created Vertex Array Object
    return vertex_array;
}

GLuint createPlaneVertexArray()
{
    // Create vertex array object
    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    // Vertex positions
    GLuint vertex_position_buffer;
    glGenBuffers(1, &vertex_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);
    GLfloat vertices[12] = {
        -0.5, -0.5,  0.0,
         0.5, -0.5,  0.0,
         0.5,  0.5,  0.0,
        -0.5,  0.5,  0.0
    };
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_position_attrib);
    glVertexAttribPointer(app.vertex_position_attrib, 3, GL_FLOAT, false, 0, 0);

    // Vertex texture coordinates
    GLuint vertex_texcoord_buffer;
    glGenBuffers(1, &vertex_texcoord_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_texcoord_buffer);
    GLfloat texcoords[8] = {
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0
    };
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texcoords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_texcoord_attrib);
    glVertexAttribPointer(app.vertex_texcoord_attrib, 2, GL_FLOAT, false, 0, 0);

    // Faces of the triangles
    GLuint vertex_index_buffer;
    glGenBuffers(1, &vertex_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_index_buffer);
    GLushort indices[6] = {
         0,  1,  2,    0,  2,  3
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLushort), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return vertex_array;
}
