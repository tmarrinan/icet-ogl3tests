#include <iostream>
#include <cmath>
#include <string>
#include <vector>
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
#include "directory.h"
#include "glslloader.h"
#include "objloader.h"
#include "imgreader.h"
#include "textrender.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define WINDOW_TITLE "Nuclear Station (IceT)"


typedef struct GlslProgram {
    GLuint program;
    std::map<std::string,GLint> uniforms;
} GlslProgram;

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
    // FPS counter
    double frame_time_start;
    int num_frames;
    bool show_fps;
    // Frame counter
    int frame_count;
    double pixel_read_time;       // only used in IceT generic compositing
    // Model info
    std::vector<ObjLoader*> model_list;
    GLuint plane_vertex_array;
    // Rendering info
    bool color_by_rank;
    std::map<std::string, GlslProgram> glsl_program;
    glm::vec4 background_color;
    glm::vec3 camera_position;
    glm::dmat4 projection_matrix;
    glm::dmat4 view_matrix;
    glm::dmat4 model_matrix;
    glm::dmat3 normal_matrix;
    glm::dmat4 composite_mv_matrix;
    GLuint text_background_texture;
    GLuint text_texture;
    glm::dmat4 text_background_mv_matrix;
    glm::dmat4 text_mv_matrix;
    double rotate_y;
    double render_time;
    GLuint vertex_position_attrib;
    GLuint vertex_normal_attrib;
    GLuint vertex_texcoord_attrib;
    GLuint composite_texture;
    TR_FontFace *font;
    GLuint framebuffer;           // only used in IceT generic compositing
    GLuint framebuffer_texture;   // only used in IceT generic compositing
    GLuint framebuffer_depth;     // only used in IceT generic compositing
    // Output to PPM image
    std::string outfile;
} AppData;

const float RANK_COLORS[16][3] = {
    {0.502, 0.000, 0.000},  // maroon   #800000
    {0.502, 0.502, 0.000},  // olive    #808000
    {0.275, 0.600, 0.561},  // teal     #469990
    {0.000, 0.000, 0.459},  // navy     #000075
    {0.902, 0.098, 0.294},  // red      #E6194B
    {0.961, 0.510, 0.192},  // orange   #F58231
    {1.000, 0.882, 0.098},  // yellow   #FFE119
    {0.749, 0.937, 0.271},  // lime     #BFEF45
    {0.235, 0.706, 0.294},  // green    #3CB44B
    {0.259, 0.831, 0.957},  // cyan     #42D4F4
    {0.263, 0.388, 0.847},  // blue     #4363D8
    {0.569, 0.118, 0.706},  // purple   #911EB4
    {0.980, 0.745, 0.831},  // pink     #FABED4
    {1.000, 0.847, 0.694},  // apricot  #FFD8B1
    {1.000, 0.980, 0.784},  // beige    #FFFAC8
    {0.863, 0.745, 1.000}   // lavender #DCBEFF
};


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
void mat3ToFloatArray(glm::dmat3 mat3, float array[9]);
void loadShader(std::string key, std::string shader_filename_base);
void loadObjModels(std::string model_path, float bbox[6]);
GLuint planeVertexArray();
void writePpm(const char *filename, int width, int height, const uint8_t *rgba);

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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize app
    init();

    // Main render loop
    double start_time, end_time, elapsed, compress_time, read_time, collect;
    uint16_t should_close = 0;
    uint32_t animation_frames = 1440;
    MPI_Barrier(MPI_COMM_WORLD);
    if (app.rank == 0)
    {
        start_time = MPI_Wtime();
    }
    while (!should_close)
    {
        // Render frame
        doFrame();

        // poll for user events
        glfwPollEvents();

        // check if any window has been closed
        uint16_t close_this = glfwWindowShouldClose(app.window);
        close_this |= (app.frame_count == animation_frames);
        MPI_Allreduce(&close_this, &should_close, 1, MPI_UINT16_T, MPI_SUM, MPI_COMM_WORLD);
    }
    if (app.rank == 0)
    {
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;
    }

    // AVERAGE or MAX more useful???
    icetGetDoublev(ICET_COMPRESS_TIME, &compress_time);
    MPI_Reduce(&compress_time, &collect, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    compress_time = collect / (double)app.num_proc;
#ifdef USE_ICET_OGL3
    icetGetDoublev(ICET_BUFFER_READ_TIME, &read_time);
#else
    read_time = app.pixel_read_time;
#endif
    MPI_Reduce(&read_time, &collect, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    read_time = collect / (double)app.num_proc;

    if (app.rank == 0)
    {
        double avg_fps = animation_frames / elapsed;
        double avg_compress_time = compress_time / animation_frames;
        double avg_read_time = read_time / animation_frames;
#if defined(USE_ICET_OGL3) && defined(ICET_USE_PARICOMPRESS)
        const char *composite_method = "IceT OGL3 w/ GPU Compression";
        const char *composite_method_short = "IceTOGL3-GPU";
#elif defined(USE_ICET_OGL3)
        const char *composite_method = "IceT OGL3";
        const char *composite_method_short = "IceTOGL3-CPU";
#else
        const char *composite_method = "IceT Generic";
        const char *composite_method_short = "IceTGeneric";
#endif
        char statfile[64];
        snprintf(statfile, 64, "NuclearPowerStation_%s_%dx%d_%dproc.txt", composite_method_short,
                 app.window_width, app.window_height, app.num_proc);
        FILE *fp = fopen(statfile, "w");
        fprintf(fp, "Data Set, Image Width, Image Height, Composite Method, Number of Processes\n");
        fprintf(fp, "Nuclear Power Station, %d, %d, %s, %d\n\n", app.window_width, app.window_height,
                composite_method, app.num_proc);
        fprintf(fp, "Average FPS, Average Compression Compute Time, Average Memory Transfer Time\n");
        fprintf(fp, "%.3lf, %.6lf, %.6lf\n\n", avg_fps, avg_compress_time, avg_read_time);
        fclose(fp);
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
    app.show_fps = false;
    app.color_by_rank = false;
    app.outfile = "";

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
        else if (argument == "--show-fps" || argument == "-f")
        {
            app.show_fps = true;
            i += 1;
        }
        else if (argument == "--color-by-rank" || argument == "-c")
        {
            app.color_by_rank = true;
            i += 1;
        }
        else if ((argument == "--outfile" || argument == "-o") && i < argc - 1)
        {
            app.outfile = argv[i + 1];
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

    // Initialize FPS counter
    app.frame_time_start = MPI_Wtime();
    app.num_frames = 0;

    // Initialize frame count
    app.frame_count = 0;
    
    // Initialize text renderer
    if (app.show_fps)
    {
        TR_Initialize();
        TR_CreateFontFace("resrc/fonts/OpenSans-Regular.ttf", 20, &(app.font));
        
        int text_bg_w, text_bg_h;
        uint8_t *text_bg_pixels;
        imageFileToRgba("resrc/images/bg_135x60.png", &text_bg_w, &text_bg_h, &text_bg_pixels);
        
        glGenTextures(1, &(app.text_background_texture));
        glBindTexture(GL_TEXTURE_2D, app.text_background_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_bg_w, text_bg_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_bg_pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        freeRgba(text_bg_pixels);
        
        double text_bg_width = text_bg_w;
        double text_bg_height = text_bg_h;
        app.text_background_mv_matrix = glm::translate(glm::dmat4(1.0), glm::dvec3((text_bg_width / 2.0) + 10.0,
                                                       (double)app.window_height - (text_bg_height / 2.0) - 10.0, 0.0));
        app.text_background_mv_matrix = glm::scale(app.text_background_mv_matrix, glm::dvec3(text_bg_width, text_bg_height, 1.0));
        
        
        uint32_t text_w, text_h, baseline;
        uint8_t *text_pixels;
        TR_RenderStringToTexture(app.font, "0.00 fps", true, &text_w, &text_h, &baseline, &text_pixels);
        
        glGenTextures(1, &(app.text_texture));
        glBindTexture(GL_TEXTURE_2D, app.text_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, text_w, text_h, 0, GL_RED, GL_UNSIGNED_BYTE, text_pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        free(text_pixels);
        
        double text_width = text_w;
        double text_height = text_h;
        app.text_mv_matrix = glm::translate(glm::dmat4(1.0), glm::dvec3((text_width / 2.0) + 28.0,
                                            (double)app.window_height - (text_height / 2.0) - baseline - 16.0, 0.1));
        app.text_mv_matrix = glm::scale(app.text_mv_matrix, glm::dvec3(text_width, text_height, 1.0));
    }

    // Set background color
    if (app.color_by_rank)
    {
        app.background_color = glm::vec4(0.85, 0.85, 0.85, 1.00);
    }
    else
    {
        app.background_color = glm::vec4(0.61, 0.84, 0.94, 1.00);
    }

    // Create projection and view matrices
    app.projection_matrix = glm::perspective(glm::radians(60.0), (double)app.window_width / (double)app.window_height, 0.1, 250.0);
    app.camera_position = glm::vec3(0.5, 2.8, -10.0);
    app.view_matrix = glm::lookAt(app.camera_position, glm::vec3(0.5, 1.7, 0.0), glm::vec3(0.0, 1.0, 0.0));
    app.model_matrix = glm::dmat4(1.0);
    app.composite_mv_matrix = glm::translate(glm::dmat4(1.0), glm::dvec3((double)app.window_width / 2.0, (double)app.window_height / 2.0, -0.5));
    app.composite_mv_matrix = glm::scale(app.composite_mv_matrix, glm::dvec3((double)app.window_width, (double)app.window_height, 1.0));

    // Set OpenGL settings
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, app.window_width, app.window_height);

    // Initialize vertex attributes
    app.vertex_position_attrib = 0;
    app.vertex_normal_attrib = 1;
    app.vertex_texcoord_attrib = 2;

    // Load shader programs
    loadShader("color", "resrc/shaders/color");
    loadShader("texture", "resrc/shaders/texture");
    loadShader("nolight", "resrc/shaders/nolight_texture");
    loadShader("text", "resrc/shaders/text");

    // Load nuclear station OBJ models
    float bbox[6];
    loadObjModels("resrc/data/nuclear_station_models", bbox);
#ifdef USE_ICET_OGL3
    icetBoundingBoxf(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
#endif

    // Initialize rotations and animation time
    app.rotate_y = 180.0;
    if (app.rank == 0)
    {
        app.render_time = MPI_Wtime();
    }
    MPI_Bcast(&(app.render_time), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Create composite texture (for display of final image)
    if (app.rank == 0)
    {
        app.plane_vertex_array = planeVertexArray();

        glGenTextures(1, &(app.composite_texture));
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Lights
    glm::vec3 ambient = glm::vec3(0.2, 0.2, 0.2);
    glm::vec3 point_light_pos = glm::vec3(0.5, 6.0, -18.0);
    glm::vec3 point_light_col = glm::vec3(1.0, 1.0, 1.0);
    glm::vec2 point_light_atten = glm::vec2(32.0, 64.0);

    // Upload static uniforms
    glUseProgram(app.glsl_program["color"].program);
    glUniform3fv(app.glsl_program["color"].uniforms["camera_position"], 1, glm::value_ptr(app.camera_position));
    glUniform3fv(app.glsl_program["color"].uniforms["light_ambient"], 1, glm::value_ptr(ambient));
    glUniform1i(app.glsl_program["color"].uniforms["num_lights"], 1);
    glUniform3fv(app.glsl_program["color"].uniforms["light_position[0]"], 1, glm::value_ptr(point_light_pos));
    glUniform3fv(app.glsl_program["color"].uniforms["light_color[0]"], 1, glm::value_ptr(point_light_col));
    glUniform2fv(app.glsl_program["color"].uniforms["light_attenuation[0]"], 1, glm::value_ptr(point_light_atten));
    glUniform1i(app.glsl_program["color"].uniforms["num_spotlights"], 0);
    
    glUseProgram(app.glsl_program["texture"].program);
    glUniform3fv(app.glsl_program["texture"].uniforms["camera_position"], 1, glm::value_ptr(app.camera_position));
    glUniform3fv(app.glsl_program["texture"].uniforms["light_ambient"], 1, glm::value_ptr(ambient));
    glUniform1i(app.glsl_program["texture"].uniforms["num_lights"], 1);
    glUniform3fv(app.glsl_program["texture"].uniforms["light_position[0]"], 1, glm::value_ptr(point_light_pos));
    glUniform3fv(app.glsl_program["texture"].uniforms["light_color[0]"], 1, glm::value_ptr(point_light_col));
    glUniform2fv(app.glsl_program["texture"].uniforms["light_attenuation[0]"], 1, glm::value_ptr(point_light_atten));
    glUniform1i(app.glsl_program["texture"].uniforms["num_spotlights"], 0);

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
    double now;
    if (app.rank == 0)
    {
        now = MPI_Wtime();

        // Update FPS every 500 milliseconds
        if (app.show_fps && now - app.frame_time_start > 0.5)
        {
            //printf("FPS: %.1lf\n", (double)app.num_frames / (now - app.frame_time_start));
            uint32_t text_w, text_h, baseline;
            uint8_t *text_pixels;
            char fps_str[16];
            snprintf(fps_str, 16, "%.2lf fps", (double)app.num_frames / (now - app.frame_time_start));
            TR_RenderStringToTexture(app.font, fps_str, true, &text_w, &text_h, &baseline, &text_pixels);
            
            glBindTexture(GL_TEXTURE_2D, app.text_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, text_w, text_h, 0, GL_RED, GL_UNSIGNED_BYTE, text_pixels);
            glBindTexture(GL_TEXTURE_2D, 0);
            
            free(text_pixels);
            
            double text_width = text_w;
            double text_height = text_h;
            app.text_mv_matrix = glm::translate(glm::dmat4(1.0), glm::dvec3((text_width / 2.0) + 28.0,
                                                (double)app.window_height - (text_height / 2.0) - baseline - 16.0, 0.1));
            app.text_mv_matrix = glm::scale(app.text_mv_matrix, glm::dvec3(text_width, text_height, 1.0));


            app.frame_time_start = now;
            app.num_frames = 0;
        }
    }
    MPI_Bcast(&now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //double dt = now - app.render_time;
    //app.rotate_y -= 15.0 * dt;
    app.rotate_y -= 0.25;
    app.model_matrix = glm::rotate(glm::dmat4(1.0), glm::radians(app.rotate_y), glm::dvec3(0.0, 1.0, 0.0));
    app.normal_matrix = glm::inverse(app.model_matrix);
    app.normal_matrix = glm::transpose(app.normal_matrix);

    app.render_time = now;

    app.num_frames++;
    app.frame_count++;
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

    // Deselect app's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Copy image to IceT buffer
    glFinish();

    double start = MPI_Wtime();

    IceTUByte *pixels = icetImageGetColorub(result);
    IceTFloat *depth = icetImageGetDepthf(result);

    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_depth);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    glBindTexture(GL_TEXTURE_2D, 0);

    double end = MPI_Wtime();
    app.pixel_read_time += end - start;
}

void render()
{
    // Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float mat4_proj[16], mat4_model[16], mat3_norm[9], mat4_view[16];
    mat4ToFloatArray(app.projection_matrix, mat4_proj);
    mat4ToFloatArray(app.view_matrix, mat4_view);
    mat4ToFloatArray(app.model_matrix, mat4_model);
    mat3ToFloatArray(app.normal_matrix, mat3_norm);

    glUseProgram(app.glsl_program["color"].program);
    glUniformMatrix4fv(app.glsl_program["color"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.glsl_program["color"].uniforms["model_matrix"], 1, GL_FALSE, mat4_model);
    glUniformMatrix3fv(app.glsl_program["color"].uniforms["normal_matrix"], 1, GL_FALSE, mat3_norm);
    glUniformMatrix4fv(app.glsl_program["color"].uniforms["view_matrix"], 1, GL_FALSE, mat4_view);
    glUseProgram(app.glsl_program["texture"].program);
    glUniformMatrix4fv(app.glsl_program["texture"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.glsl_program["texture"].uniforms["model_matrix"], 1, GL_FALSE, mat4_model);
    glUniformMatrix3fv(app.glsl_program["texture"].uniforms["normal_matrix"], 1, GL_FALSE, mat3_norm);
    glUniformMatrix4fv(app.glsl_program["texture"].uniforms["view_matrix"], 1, GL_FALSE, mat4_view);
    glUseProgram(0);

    int i, j;
    for (i = 0; i < app.model_list.size(); i++)
    {
        std::vector<Model> models = app.model_list[i]->getModelList();
        for (j = 0; j < models.size(); j++)
        {
            if (app.color_by_rank)
            {
                glUseProgram(app.glsl_program["color"].program);
                std::string program_name = "color";
                Material mat = app.model_list[i]->getMaterial(models[j].material_name);

                glUniform3fv(app.glsl_program[program_name].uniforms["material_color"], 1, RANK_COLORS[app.rank]);
                glUniform3fv(app.glsl_program[program_name].uniforms["material_specular"], 1, glm::value_ptr(mat.specular));
                glUniform1f(app.glsl_program[program_name].uniforms["material_shininess"], mat.shininess);
            }
            else
            {
                std::string program_name;
                Material mat = app.model_list[i]->getMaterial(models[j].material_name);

                if (mat.has_texture)
                {
                    glUseProgram(app.glsl_program["texture"].program);
                    program_name = "texture";

                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, mat.texture_id);
                    glUniform1i(app.glsl_program[program_name].uniforms["image"], 0);
                }
                else
                {
                    glUseProgram(app.glsl_program["color"].program);
                    program_name = "color";
                }

                glUniform3fv(app.glsl_program[program_name].uniforms["material_color"], 1, glm::value_ptr(mat.color));
                glUniform3fv(app.glsl_program[program_name].uniforms["material_specular"], 1, glm::value_ptr(mat.specular));
                glUniform1f(app.glsl_program[program_name].uniforms["material_shininess"], mat.shininess);
            }
            glBindVertexArray(models[j].vertex_array);
            glDrawElements(GL_TRIANGLES, models[j].face_index_count, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }

    glUseProgram(0);
}

void display()
{
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.rank == 0)
    {
        glUseProgram(app.glsl_program["nolight"].program);
        
        float mat4_projection[16], mat4_modelview[16];
        mat4ToFloatArray(glm::ortho(0.0, (double)app.window_width, 0.0, (double)app.window_height, -1.0, 1.0), mat4_projection);
        mat4ToFloatArray(app.composite_mv_matrix, mat4_modelview);
        glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_projection);
        glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["modelview_matrix"], 1, GL_FALSE, mat4_modelview);
   
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        IceTUByte *pixels = icetImageGetColorub(app.image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.window_width, app.window_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glUniform1i(app.glsl_program["nolight"].uniforms["image"], 0);

        glBindVertexArray(app.plane_vertex_array);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        
        if (app.show_fps)
        {
            mat4ToFloatArray(app.text_background_mv_matrix, mat4_modelview);
            glBindTexture(GL_TEXTURE_2D, app.text_background_texture);
            glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["modelview_matrix"], 1, GL_FALSE, mat4_modelview);
            
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
            
            
            glUseProgram(app.glsl_program["text"].program);
            
            mat4ToFloatArray(app.text_mv_matrix, mat4_modelview);
            glUniformMatrix4fv(app.glsl_program["text"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_projection);
            glUniformMatrix4fv(app.glsl_program["text"].uniforms["modelview_matrix"], 1, GL_FALSE, mat4_modelview);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, app.text_texture);
            glUniform1i(app.glsl_program["text"].uniforms["image"], 0);
            
            float white[3] = {1.0, 1.0, 1.0};
            glUniform3fv(app.glsl_program["text"].uniforms["font_color"], 1, white);
            
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        }
        
        glBindVertexArray(0);

        glUseProgram(0);

        if (app.outfile != "")
        {
            int i;
            char filename[128];
            snprintf(filename, 128, "%s_%05d.ppm", app.outfile.c_str(), app.frame_count);
            FILE *fp = fopen(filename, "wb");
            fprintf(fp, "P6\n%d %d\n255\n", app.window_width, app.window_height);
            for (i = 0; i < app.window_width * app.window_height; i++)
            {
                fwrite(pixels + (4 * i), 3, 1, fp);
            }
            fclose(fp);
        }
    }
    
    // Synchronize and display
    MPI_Barrier(MPI_COMM_WORLD);
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

void mat3ToFloatArray(glm::dmat3 mat3, float array[9])
{
    array[0] = mat3[0][0];
    array[1] = mat3[0][1];
    array[2] = mat3[0][2];
    array[3] = mat3[1][0];
    array[4] = mat3[1][1];
    array[5] = mat3[1][2];
    array[6] = mat3[2][0];
    array[7] = mat3[2][1];
    array[8] = mat3[2][2];
}

void loadShader(std::string key, std::string shader_filename_base)
{
    // Compile GPU program
    GlslProgram p;
    std::string vert_filename = shader_filename_base + ".vert";
    std::string frag_filename = shader_filename_base + ".frag";
    p.program = glsl::createShaderProgram(vert_filename.c_str(), frag_filename.c_str());

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(p.program, app.vertex_position_attrib, "vertex_position");
    glBindAttribLocation(p.program, app.vertex_normal_attrib, "vertex_normal");
    glBindAttribLocation(p.program, app.vertex_texcoord_attrib, "vertex_texcoord");
    glBindFragDataLocation(p.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(p.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(p.program, p.uniforms);

    // Store GPU program and uniforms
    app.glsl_program[key] = p;
}

void loadObjModels(std::string model_path, float bbox[6])
{
    bbox[0] =  9.9e12; // x min
    bbox[1] = -9.9e12; // x max
    bbox[2] =  9.9e12; // y min
    bbox[3] = -9.9e12; // y max
    bbox[4] =  9.9e12; // z min
    bbox[5] = -9.9e12; // z max
    std::vector<std::string> obj_filenames = directory::listFiles(model_path, "obj");
    
    int i;
    uint32_t total_triangles = 0;
    for (i = app.rank; i < obj_filenames.size(); i += app.num_proc)
    {
        std::string obj_path = model_path + "/" + obj_filenames[i];
        ObjLoader *model = new ObjLoader(obj_path.c_str());

        glm::vec3 center = model->getCenter();
        glm::vec3 size = model->getSize();
        if (center[0] - (size[0] / 2.0) < bbox[0])
        {
            bbox[0] = center[0] - (size[0] / 2.0);
        }
        if (center[0] + (size[0] / 2.0) > bbox[1])
        {
            bbox[1] = center[0] + (size[0] / 2.0);
        }
        if (center[1] - (size[1] / 2.0) < bbox[2])
        {
            bbox[2] = center[1] - (size[1] / 2.0);
        }
        if (center[1] + (size[1] / 2.0) > bbox[3])
        {
            bbox[3] = center[1] + (size[1] / 2.0);
        }
        if (center[2] - (size[2] / 2.0) < bbox[4])
        {
            bbox[4] = center[2] - (size[2] / 2.0);
        }
        if (center[2] + (size[2] / 2.0) > bbox[5])
        {
            bbox[5] = center[2] + (size[2] / 2.0);
        }
        total_triangles += model->getNumberOfTriangles();
        app.model_list.push_back(model);
    }

    printf("[rank % 2d]: %u triangles\n", app.rank, total_triangles);
}

GLuint planeVertexArray()
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

    return vertex_array;
}

void writePpm(const char *filename, int width, int height, const uint8_t *rgba)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: could not create file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    int i;
    for (i = 0; i < width * height; i++)
    {
        fprintf(fp, "%c%c%c", rgba[i * 4 + 0], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    }

    fclose(fp);
}
