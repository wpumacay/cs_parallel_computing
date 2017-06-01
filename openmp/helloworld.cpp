
#include <iostream>
#include <omp.h>

int main()
{
    
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        std::cout << "hello" << ID << " ";
        std::cout << "world" << ID << std::endl;
    }


    return 0;
}
