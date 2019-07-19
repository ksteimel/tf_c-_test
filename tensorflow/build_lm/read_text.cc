#include <sstream>
#include <iostream>
#include <locale>
#include <fstream>
#include <codecvt>
#include <map>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

std::map<std::wstring, int> generateWordMapping(std::string filepath)
{
    std::ios::sync_with_stdio(false);
    std::locale loc("en_US.UTF-8");
    std::wcout.imbue(loc);
    std::wfstream file(filepath);
    file.imbue(loc);
    std::map<std::wstring, int> wordToIndex;
    int index = 0;
    for (std::wstring instr; std::getline(file, instr); index++) {
        wordToIndex[instr] = index;
    }
    return wordToIndex;
}

Tensor readCheckpoint(std::string pathToGraph, std::string pathToCheckpoint)
{
    auto session = NewSession(SessionOptions());
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

    // Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

    // Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    // and run the inference to your liking
    //auto feedDict = ...
    //auto outputOps = ...
    //std::vector<tensorflow::Tensor> outputTensors;
    //status = session->Run(feedDict, outputOps, {}, &outputTensors);
}
int main(void)
{
   auto wordToIndex = generateWordMapping("labels.txt"); 
}
