// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/types.h>

namespace SOFT {

#if defined(WITH_CUDA) || defined(WITH_HIP)
    void GaussianSubtraction_forward_cuda(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output
        );

    void GaussianSubtraction_backward_query_cuda(
        const at::Tensor query,
        const at::Tensor key,
        const at::Tensor gradOutput,
        at::Tensor gradQuery
        );

    void GaussianSubtraction_backward_key_cuda(
        const at::Tensor query,
        const at::Tensor key,
        const at::Tensor gradOutput,
        at::Tensor gradKey
        );

    void GaussianSubtraction_reduce_forward_cuda(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output
        );
#endif

// Interface for Python
inline void GaussianSubtraction_forward(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output) {
    if (query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(query.is_cuda(), "query tensor is not on GPU!");
    TORCH_CHECK(key.is_cuda(), "key tensor is not on GPU!");
    return GaussianSubtraction_forward_cuda(
            query,
            key,
            output
            );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


inline void GaussianSubtraction_backward_query(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradQuery) {
    if (gradOutput.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(query.is_cuda(), "query tensor is not on GPU!");
    TORCH_CHECK(key.is_cuda(), "key tensor is not on GPU!");
    return GaussianSubtraction_backward_query_cuda(
            gradOutput,
            query,
            key,
            gradQuery
            );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}


inline void GaussianSubtraction_backward_key(
        const at::Tensor gradOutput,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor gradKey) {
    if (gradOutput.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(query.is_cuda(), "query tensor is not on GPU!");
    TORCH_CHECK(key.is_cuda(), "key tensor is not on GPU!");
    return GaussianSubtraction_backward_key_cuda(
            gradOutput,
            query,
            key,
            gradKey
            );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

inline void GaussianSubtraction_reduce_forward(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output) {
    if (query.is_cuda() && key.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    TORCH_CHECK(query.is_cuda(), "query tensor is not on GPU!");
    TORCH_CHECK(key.is_cuda(), "key tensor is not on GPU!");
    return GaussianSubtraction_reduce_forward_cuda(
            query,
            key,
            output
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

} // namespace SOFT
