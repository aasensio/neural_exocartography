#ifndef NOISE_SIMPLEGENERATOR_H
#define NOISE_SIMPLEGENERATOR_H

#include <iostream>
#include <sstream>
#include "Generator.h"

class SimpleGenerator : public Generator {
public:
    void generate(int seed);
};

#endif
