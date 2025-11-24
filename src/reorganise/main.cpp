#include "Application.h"
#include <iostream>

int main(int argc, char *argv[])
{
    try
    {
        Application app;
        if (!app.initialize())
        {
            std::cerr << "Failed to initialize application" << std::endl;
            return 1;
        }

        app.run();
        std::cout << "Exiting successfully" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



