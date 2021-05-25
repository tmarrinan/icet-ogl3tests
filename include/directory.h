#ifndef DIRECTORY_H
#define DIRECTORY_H

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
#else
#include <dirent.h>
#endif
#include <iostream>
#include <string>
#include <vector>

namespace directory {
    std::vector<std::string> listFiles(std::string dir_path, std::string ext = "");
}


#endif // DIRECTORY_H
