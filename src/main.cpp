#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "xxhash.h"
using namespace torch;

template <typename I> std::string n2hexstr(I w, size_t hex_len = sizeof(I)<<1) {
    static const char* digits = "0123456789ABCDEF";
    std::string rc(hex_len,'0');
    for (size_t i=0, j=(hex_len-1)*4 ; i<hex_len; ++i,j-=4)
        rc[i] = digits[(w>>j) & 0x0f];
    return rc;
}

std::string calcul_hash_streaming_xxh3_128(torch::Tensor input_tensor){
    TORCH_CHECK(input_tensor.device().is_cpu(), "input_tensor must be a CPU tensor");

    size_t size = input_tensor.numel() * input_tensor.element_size();

    XXH64_hash_t seed = 0;

    // Calculate hash as 128 bits XXH3
    XXH128_hash_t hash = XXH3_128bits_withSeed(reinterpret_cast<unsigned char*>(input_tensor.data_ptr()),size,seed);
    
    std::string hash_str = n2hexstr<uint64_t>(hash.high64) + n2hexstr<uint64_t>(hash.low64);
    return hash_str;
}


std::string calcul_hash_streaming_xxh3_32(torch::Tensor input_tensor){
    TORCH_CHECK(input_tensor.device().is_cpu(), "input_tensor must be a CPU tensor");
    size_t size = input_tensor.numel() * input_tensor.element_size();

    XXH64_hash_t seed = 0;

    XXH32_hash_t hash = XXH32(reinterpret_cast<unsigned char*>(input_tensor.data_ptr()),size,seed);
    return n2hexstr<uint32_t>(uint32_t(hash));
}

std::string calcul_hash_streaming_xxh3_64(torch::Tensor input_tensor){
    TORCH_CHECK(input_tensor.device().is_cpu(), "input_tensor must be a CPU tensor");
    // Get size of tensor
    size_t size = input_tensor.numel() * input_tensor.element_size();

    XXH64_hash_t seed = 0;
    // Calculate hash as 64 bits XXH3
    XXH64_hash_t hash = XXH3_64bits_withSeed(reinterpret_cast<unsigned char*>(input_tensor.data_ptr()),size,seed);

    return n2hexstr<uint64_t>(uint64_t(hash));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Fast
    m.def("calc_hash_64", &calcul_hash_streaming_xxh3_64, "Calcul hash 64 bits");
    // Slow
    m.def("calc_hash_32", &calcul_hash_streaming_xxh3_32, "Calcul hash 32 bits");
    // Fast
    m.def("calc_hash_128", &calcul_hash_streaming_xxh3_128, "Calcul hash 128 bits");
}