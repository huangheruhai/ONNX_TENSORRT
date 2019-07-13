#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
using namespace nvinfer1;

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

samplesCommon::Args gArgs;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();

    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex{}, outputIndex{};
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};

    if (!onnxToTRTModel("mnist.onnx", 1, trtModelStream))
        gLogger.reportFail(sampleTest);

    assert(trtModelStream != nullptr);

    // read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H * INPUT_W];
    int num = rand() % 10;
    readPGMFile(locateFile(std::to_string(num) + ".pgm", gArgs.dataDirs), fileData);

    // print an ascii representation
    gLogInfo << "Input:\n";
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    gLogInfo << std::endl;

    float data[INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        data[i] = 1.0 - float(fileData[i] / 255.0);

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    // run inference
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    float val{0.0f};
    int idx{0};

    //Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] = exp(prob[i]);
        sum += prob[i];
    }

    gLogInfo << "Output:\n";
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] /= sum;
        val = std::max(val, prob[i]);
        if (val == prob[i])
            idx = i;

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    gLogInfo << std::endl;

    bool pass{idx == num && val > 0.9f};

    return gLogger.reportTest(sampleTest, pass);
}
