#include <sstream>
#include <iostream>
#include <locale>
#include <fstream>
#include <codecvt>
#include <map>
#include <io_ops.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

using namespace std;
using namespace tensorflow;

std::map<std::string, int> generateWordMapping(std::string filepath)
{
    std::ios::sync_with_stdio(false);
    std::locale loc("en_US.UTF-8");
    std::cout.imbue(loc);
    std::fstream file(filepath);
    file.imbue(loc);
    std::map<std::string, int> wordToIndex;
    int index = 0;
    for (std::string instr; std::getline(file, instr); index++) {
        wordToIndex[instr] = index;
    }
    return wordToIndex;
}

ClientSession readCheckpoint(std::string pathToGraph, std::string pathToCheckpoint)
{
    Scope root = Scope::NewRootScope();
    auto session = ClientSession(root);
    //if (session == nullptr) {
    //    throw runtime_error("Could not create Tensorflow session.");
    //}

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
    checkpointPathTensor.scalar<std::string>()() = pathToCheckpoint;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + pathToCheckpoint + ": " + status.ToString());
    }
    return session;
    // Run inference
    //auto feedDict = ...
    //auto outputOps = ...
    //std::vector<tensorflow::Tensor> outputTensors;
    //status = session->Run(feedDict, outputOps, {}, &outputTensors);
}
// simple string splitting utliity
std::vector<std::string> split(std::string& input, char delim = ' ')
{
    std::vector<std::string> splitOutput;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, delim))
    {
        splitOutput.push_back(token);
    }
    return splitOutput;
}

// Change a sentence (represented as a string) into a sequence of integers separated by the 
// The delimiter specified is used to split the long string into a sequence of strings
// and then each string in the sequence is replaced with the appropriate integer substitute
// All sentence sequences are terminated by an end of sentence marker.
/*std::vector<int> sentToSeq(std::wstring inputSent, std::wchar delim, std::map<std::wstring, int> wordToIndex)
{
    
}*/
/*
// prime model with a sequence of characters
// this runs the data through the model element by element, so as to update its internal state (stored in t_state)
// next time we feed the model an element to make a prediction, it will make the prediction primed on this state (i.e. sequence of elements)
void prime_model(string prime_data, int prime_length) {
    t_state = tensorflow::Tensor(); // reset initial state to use zeros
    for(int i=MAX(0, prime_data.size()-prime_length); i<prime_data.size(); i++) {
        run_model(prime_data[i], t_state);
    }
}

void run_model(char ch, const tensorflow::Tensor &state_in = tensorflow::Tensor()) {
    // copy input data into tensor
    msa::tf::scalar_to_tensor(char_to_int[ch], t_data_in);

    // run graph, feed inputs, fetch output
    vector<string> fetch_tensors = { "data_out", "state_out" };
    tensorflow::Status status;
    if(state_in.NumElements() > 0) {
        // use state_in if passed in as parameter
        status = session->Run({ { "data_in", t_data_in }, { "state_in", state_in } }, fetch_tensors, {}, &t_out);
    } else {
        // otherwise model will use internally init state to zeros
        status = session->Run({ { "data_in", t_data_in }}, fetch_tensors, {}, &t_out);
    }

    if(status != tensorflow::Status::OK()) {
        ofLogError() << status.error_message();
        return;
    }

    // convert model output from tensors to more manageable types
    if(t_out.size() > 1) {
        last_model_output = msa::tf::tensor_to_vector<float>(t_out[0]);
        last_model_output = msa::tf::adjust_probs_with_temp(last_model_output, sample_temp);

        // save lstm state for next run
        t_state = t_out[1];
    }
}
float getProbability(std::wstring inputSent, ClientSession session)
{
    
}*/
int main(void)
{
   auto wordToIndex = generateWordMapping("labels.txt"); 
   std::string instr = "Hello this is a test";
   auto splits = split(instr, ' ');
   std::cout << splits[0] << " " << splits[1] << " " << splits[2] << std::endl;
}
