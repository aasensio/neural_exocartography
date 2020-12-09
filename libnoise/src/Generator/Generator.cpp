#include "Generator.h"

using namespace std;

Generator::Generator () {
    this->writer = utils::WriterBMP();
    this->image  = utils::Image();
}

Generator::~Generator () {
}

string Generator::createSurfaceFilename(int seed) {
    stringstream filename;
    filename << "planet-" << seed << "-surface.bmp";
    return filename.str();
}

string Generator::createNormalMapFilename(int seed) {
    stringstream filename;
    filename << "planet-" << seed << "-normal.bmp";
    return filename.str();
}

string Generator::createSpecularMapFilename(int seed) {
    stringstream filename;
    filename << "planet-" << seed << "-specular.bmp";
    return filename.str();
}

void Generator::writeSurface(string filename, const utils::NoiseMap & noiseMap) {
    utils::RendererImage renderer;
    renderer.SetSourceNoiseMap(noiseMap);
    renderer.SetDestImage(this->image);
    renderer.ClearGradient();
    renderer.AddGradientPoint(-1.0000, utils::Color(  0,   0, 128, 255)); // deeps
    renderer.AddGradientPoint(-0.2500, utils::Color(  0,   0, 255, 255)); // shallow
    renderer.AddGradientPoint( 0.0000, utils::Color(  0, 128, 255, 255)); // shore
    renderer.AddGradientPoint( 0.0625, utils::Color(240, 240,  64, 255)); // sand
    renderer.AddGradientPoint( 0.1250, utils::Color( 32, 160,   0, 255)); // grass
    renderer.AddGradientPoint( 0.2500, utils::Color( 60, 140,  40, 255)); // forest
    renderer.AddGradientPoint( 0.7500, utils::Color(128, 128, 128, 255)); // rock
    renderer.AddGradientPoint( 1.0000, utils::Color(255, 255, 255, 255)); // snow
    renderer.EnableLight();
    renderer.SetLightContrast(3.0);
    renderer.SetLightBrightness(2.0);
    renderer.Render();

    this->writer.SetSourceImage(this->image);
    this->writer.SetDestFilename(filename);
    this->writer.WriteDestFile();
}

void Generator::writeSpecularMap(string filename, const utils::NoiseMap & noiseMap) {
    const double SEA_LEVEL = 0.0;
    const double MAX_ELEV  = 8192.0;
    const double MIN_ELEV  = -8192.0;
    double seaLevelInMeters = (((SEA_LEVEL + 1.0) / 2.0) * (MAX_ELEV - MIN_ELEV)) + MIN_ELEV;

    utils::RendererImage renderer;
    renderer.SetSourceNoiseMap(noiseMap);
    renderer.SetDestImage(this->image);
    renderer.ClearGradient();
    renderer.AddGradientPoint(MIN_ELEV              , utils::Color (255, 255, 255, 255));
    renderer.AddGradientPoint(seaLevelInMeters      , utils::Color (255, 255, 255, 255));
    renderer.AddGradientPoint(seaLevelInMeters + 1.0, utils::Color (0, 0, 0, 255));
    renderer.AddGradientPoint(MAX_ELEV              , utils::Color (128, 128, 128, 255));
    renderer.EnableLight(false);
    renderer.Render();

    this->writer.SetSourceImage(this->image);
    this->writer.SetDestFilename(filename);
    this->writer.WriteDestFile();
}

void Generator::writeNormalMap(string filename, const utils::NoiseMap & noiseMap) {
    int EAST_COORD              = 180;
    int WEST_COORD              = -180;
    double GRID_WIDTH           = 1024;
    double PLANET_CIRCUMFERENCE = 44236800.0;
    double degExtent       = EAST_COORD - WEST_COORD;
    double metersPerDegree = (PLANET_CIRCUMFERENCE / 360.0);
    double resInMeters     = (degExtent / GRID_WIDTH) * metersPerDegree;

    utils::RendererNormalMap renderer;
    renderer.SetSourceNoiseMap(noiseMap);
    renderer.SetDestImage(this->image);
    renderer.SetBumpHeight(1.0 / resInMeters);
    renderer.Render();

    this->writer.SetSourceImage(this->image);
    this->writer.SetDestFilename(filename);
    this->writer.WriteDestFile();
}
