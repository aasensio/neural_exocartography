#include "main.h"

using namespace std;

int main (int argc, char ** argv)
{
    if (argc < 3) {
        cout << "Missing arguments: [simple|complex] [seed]" << endl;
        return 1;
    }
    string type(argv[1]);
    string seed(argv[2]);

    Generator * generator = nullptr;
    if (type == "simple") {
        generator = new SimpleGenerator();
    } else {
        generator = new ComplexGenerator();
    }
    generator->generate(stoi(seed));
    generator->~Generator();

    return 0;
}
