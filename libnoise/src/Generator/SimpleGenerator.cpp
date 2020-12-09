#include "SimpleGenerator.h"

using namespace std;

void SimpleGenerator::generate(int seed) {
    module::Perlin perlin;
    perlin.SetSeed(seed);

    utils::NoiseMap heightMap;
    utils::NoiseMapBuilderSphere heightMapBuilder;
    heightMapBuilder.SetSourceModule(perlin);
    heightMapBuilder.SetDestNoiseMap(heightMap);
    heightMapBuilder.SetDestSize(1024, 512);
    heightMapBuilder.SetBounds(-90.0, 90.0, -180.0, 180.0);
    heightMapBuilder.Build();

    string filename = "";

    filename = this->createSurfaceFilename(seed);
    this->writeSurface(filename, heightMap);

    filename = this->createSpecularMapFilename(seed);
    this->writeSpecularMap(filename, heightMap);

    filename = this->createNormalMapFilename(seed);
    this->writeNormalMap(filename, heightMap);
}
