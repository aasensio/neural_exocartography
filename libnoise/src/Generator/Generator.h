#ifndef NOISE_GENERATOR_H
#define NOISE_GENERATOR_H

#include "./../libnoise/noise/noise.h"
#include "./../noiseutils/noiseutils.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace utils;

class Generator {
public:
    Generator();
    ~Generator();
    virtual void generate(int seed) = 0;

protected:
    utils::WriterBMP writer;
    utils::Image image;

    string createSurfaceFilename(int seed);
    string createNormalMapFilename(int seed);
    string createSpecularMapFilename(int seed);

    void writeSurface(string filename, const utils::NoiseMap & noiseMap);
    void writeSpecularMap(string filename, const utils::NoiseMap & noiseMap);
    void writeNormalMap(string filename, const utils::NoiseMap & noiseMap);
};

#endif
