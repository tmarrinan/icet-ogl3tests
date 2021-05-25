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

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define WINDOW_TITLE "Nuclear Station (IceT)"


typedef struct GlslProgram {
    GLuint program;
    std::map<std::string,GLint> uniforms;
} GlslProgram;

typedef struct ObjModel {
    ObjLoader *model;
    glm::mat4 mv_matrix;
    glm::mat3 norm_matrix;
} ObjModel;

typedef struct AppData {
    // OpenGL window
    int window_width;
    int window_height;
    GLFWwindow *window;
    // MPI info
    int rank;
    int num_proc;
    // FPS counter
    double frame_time_start;
    int num_frames;
    // IceT info
    IceTCommunicator comm;
    IceTContext context;
    IceTImage image;
    // Model info
    std::vector<ObjModel> model_list;
    GLuint plane_vertex_array;
    // Rendering info
    bool color_by_rank;
    std::map<std::string, GlslProgram> glsl_program;
    glm::vec4 background_color;
    glm::vec3 camera_position;
    glm::dmat4 projection_matrix;
    glm::dmat4 view_matrix;
    double rotate_y;
    double render_time;
    GLuint vertex_position_attrib;
    GLuint vertex_normal_attrib;
    GLuint vertex_texcoord_attrib;
    GLuint composite_texture;
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
void render(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
            const IceTInt *readback_viewport, IceTUInt framebuffer_id);
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
    doFrame();
    uint16_t should_close = 0;
    while (!should_close)
    {
        // poll for user events
        glfwPollEvents();

        // check if any window has been closed
        uint16_t close_this = glfwWindowShouldClose(app.window);
        MPI_Allreduce(&close_this, &should_close, 1, MPI_UINT16_T, MPI_SUM, MPI_COMM_WORLD);

        // render next frame
        doFrame();
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
    app.color_by_rank = false;

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
        if (argument == "--width" || argument == "-c")
        {
            app.color_by_rank = false;
            i += 1;
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
    icetGL3Initialize();

    // Initialize FPS counter
    app.frame_time_start = MPI_Wtime();
    app.num_frames = 0;

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
    icetGL3DrawCallbackTexture(render);

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

    // Set OpenGL settings
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, app.window_width, app.window_height);

    // Initialize vertex attributes
    app.vertex_position_attrib = 0;
    app.vertex_normal_attrib = 1;
    app.vertex_texcoord_attrib = 2;

    // Load shader programs
    loadShader("color", "resrc/shaders/color");
    loadShader("texture", "resrc/shaders/texture");
    loadShader("nolight", "resrc/shaders/nolight_texture");

    // Load nuclear station OBJ models
    float bbox[6];
    loadObjModels("resrc/data/nuclear_station_models", bbox);
    icetBoundingBoxf(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

    // Initialize rotations and animation time
    app.rotate_y = 217.5;
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
    float mat4_proj[16], mat4_view[16];
    mat4ToFloatArray(app.projection_matrix, mat4_proj);
    mat4ToFloatArray(app.view_matrix, mat4_view);

    glUseProgram(app.glsl_program["color"].program);
    glUniformMatrix4fv(app.glsl_program["color"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.glsl_program["color"].uniforms["view_matrix"], 1, GL_FALSE, mat4_view);
    glUniform3fv(app.glsl_program["color"].uniforms["camera_position"], 1, glm::value_ptr(app.camera_position));
    glUniform3fv(app.glsl_program["color"].uniforms["light_ambient"], 1, glm::value_ptr(ambient));
    glUniform1i(app.glsl_program["color"].uniforms["num_lights"], 1);
    glUniform3fv(app.glsl_program["color"].uniforms["light_position[0]"], 1, glm::value_ptr(point_light_pos));
    glUniform3fv(app.glsl_program["color"].uniforms["light_color[0]"], 1, glm::value_ptr(point_light_col));
    glUniform2fv(app.glsl_program["color"].uniforms["light_attenuation[0]"], 1, glm::value_ptr(point_light_atten));
    glUniform1i(app.glsl_program["color"].uniforms["num_spotlights"], 0);
    
    glUseProgram(app.glsl_program["texture"].program);
    glUniformMatrix4fv(app.glsl_program["texture"].uniforms["projection_matrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.glsl_program["texture"].uniforms["view_matrix"], 1, GL_FALSE, mat4_view);
    glUniform3fv(app.glsl_program["texture"].uniforms["camera_position"], 1, glm::value_ptr(app.camera_position));
    glUniform3fv(app.glsl_program["texture"].uniforms["light_ambient"], 1, glm::value_ptr(ambient));
    glUniform1i(app.glsl_program["texture"].uniforms["num_lights"], 1);
    glUniform3fv(app.glsl_program["texture"].uniforms["light_position[0]"], 1, glm::value_ptr(point_light_pos));
    glUniform3fv(app.glsl_program["texture"].uniforms["light_color[0]"], 1, glm::value_ptr(point_light_col));
    glUniform2fv(app.glsl_program["texture"].uniforms["light_attenuation[0]"], 1, glm::value_ptr(point_light_atten));
    glUniform1i(app.glsl_program["texture"].uniforms["num_spotlights"], 0);

    glUseProgram(app.glsl_program["nolight"].program);
    glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["projection_matrix"], 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0)));
    glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["view_matrix"], 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0)));
    glUniformMatrix4fv(app.glsl_program["nolight"].uniforms["model_matrix"], 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0)));
    
    glUseProgram(0);
}

void doFrame()
{
    // Animation
    double now;
    if (app.rank == 0)
    {
        now = MPI_Wtime();

        // Print FPS every 2 seconds
        if (now - app.frame_time_start > 2.0)
        {
            printf("FPS: %.1lf\n", (double)app.num_frames / (now - app.frame_time_start));
            app.frame_time_start = now;
            app.num_frames = 0;
        }
    }
    MPI_Bcast(&now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double dt = now - app.render_time;
    app.rotate_y -= 15.0 * dt;

    app.view_matrix = glm::rotate(glm::mat4(1.0), (float)glm::radians(app.rotate_y), glm::vec3(0.0, 1.0, 0.0));

    int i;
    for (i = 0; i < app.model_list.size(); i++)
    {
        app.model_list[i].mv_matrix = glm::rotate(glm::mat4(1.0), (float)glm::radians(app.rotate_y), glm::vec3(0.0, 1.0, 0.0));
        app.model_list[i].norm_matrix = glm::inverse(app.model_list[i].mv_matrix);
        app.model_list[i].norm_matrix = glm::transpose(app.model_list[i].norm_matrix);
    }

    app.render_time = now;

    // Offscreen render and composit
    app.image = icetGL3DrawFrame(glm::value_ptr(app.projection_matrix),
                                 glm::value_ptr(app.view_matrix));

    // Render composited image to fullscreen quad on screen of rank 0
    display();

    app.num_frames++;

    fflush(stdout);
}

void render(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
            const IceTInt *readback_viewport, IceTUInt framebuffer_id)
{
    // Render
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int i, j;
    for (i = 0; i < app.model_list.size(); i++)
    {
        std::vector<Model> models = app.model_list[i].model->getModelList();
        for (j = 0; j < models.size(); j++)
        {
            if (app.color_by_rank)
            {
                glUseProgram(app.glsl_program["color"].program);
                std::string program_name = "color";
                Material mat = app.model_list[i].model->getMaterial(models[j].material_name);

                glUniformMatrix4fv(app.glsl_program[program_name].uniforms["model_matrix"], 1, GL_FALSE, glm::value_ptr(app.model_list[i].mv_matrix));
                glUniformMatrix3fv(app.glsl_program[program_name].uniforms["normal_matrix"], 1, GL_FALSE, glm::value_ptr(app.model_list[i].norm_matrix));
                glUniform3fv(app.glsl_program[program_name].uniforms["material_color"], 1, RANK_COLORS[app.rank]);
                glUniform3fv(app.glsl_program[program_name].uniforms["material_specular"], 1, glm::value_ptr(mat.specular));
                glUniform1f(app.glsl_program[program_name].uniforms["material_shininess"], mat.shininess);
            }
            else
            {
                std::string program_name;
                Material mat = app.model_list[i].model->getMaterial(models[j].material_name);

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

                glUniformMatrix4fv(app.glsl_program[program_name].uniforms["model_matrix"], 1, GL_FALSE, glm::value_ptr(app.model_list[i].mv_matrix));
                glUniformMatrix3fv(app.glsl_program[program_name].uniforms["normal_matrix"], 1, GL_FALSE, glm::value_ptr(app.model_list[i].norm_matrix));
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
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void display()
{
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.rank == 0)
    {
        glUseProgram(app.glsl_program["nolight"].program);
   
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        IceTUByte *pixels = icetImageGetColorub(app.image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.window_width, app.window_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glUniform1i(app.glsl_program["nolight"].uniforms["image"], 0);

        glBindVertexArray(app.plane_vertex_array);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);

        glUseProgram(0);
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
    for (i = app.rank; i < obj_filenames.size(); i += app.num_proc)
    {
        ObjModel model;
        std::string obj_path = model_path + "/" + obj_filenames[i];
        model.model = new ObjLoader(obj_path.c_str());

        glm::vec3 center = model.model->getCenter();
        glm::vec3 size = model.model->getSize();
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

        model.mv_matrix = glm::mat4(1.0);
        model.norm_matrix = glm::mat3(1.0);
        app.model_list.push_back(model);
    }
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
        -1.0, -1.0,  0.0,
         1.0, -1.0,  0.0,
         1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0
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
