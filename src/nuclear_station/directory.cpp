#include "directory.h"

std::vector<std::string> directory::listFiles(std::string dir_path, std::string ext)
{
    std::vector<std::string> files;
    
#ifdef _WIN32
    TCHAR dir_path_win[256];
    StringCchCopy(dir_path_win, 256, dir_path.c_str());
    StringCchCat(dir_path_win, 256, TEXT("\\*"));
    
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(dir_path_win, &findFileData);
    do
    {
        std::string filename = findFileData.cFileName;
        if (ext == "" || (filename.length() > ext.length() && filename.substr(filename.length() - ext.length()) == ext))
        {
            files.push_back(filename);
        }
    } while (FindNextFile(hFind, &findFileData) != 0);
    FindClose(hFind);
#else
    struct dirent *ent;
    DIR *dir = opendir(dir_path.c_str());
    if (dir != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string filename = ent->d_name;
            if (ext == "" || (filename.length() > ext.length() && filename.substr(filename.length() - ext.length()) == ext))
            {
                files.push_back(filename);
            }
        }
        closedir(dir);
    }
    else
    {
        fprintf(stderr, "Error: directory '%s' not found\n", dir_path.c_str());
    }
#endif
    
    return files;
}