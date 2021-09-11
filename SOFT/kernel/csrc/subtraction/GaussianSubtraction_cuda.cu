//
// Created by lujiachen on 2021/4/21.
//

#include <torch/types.h>

#include "GaussianSubtraction.h"
#include <stdio.h>
#include <cmath>
#include <vector>

namespace SOFT {
    void shape_check(
            at::Tensor query,
            at::Tensor key){
        TORCH_CHECK(query.ndimension() == 4,
                    "4D weight tensor (batch size, head number, sequence length, channel) expected, "
                    "but got: %s",
                    query.ndimension());
        TORCH_CHECK(key.ndimension() == 4,
                    "4D weight tensor (batch size, head number, sequence length, channel) expected, "
                    "but got: %s",
                    key.ndimension());
        TORCH_CHECK(query.is_contiguous(), "query tensor has to be contiguous");
        TORCH_CHECK(key.is_contiguous(), "key tensor has to be contiguous");
        TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(3) == key.size(2), "query and key must match their size");
    }

    void subtraction_gaussian_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels);

    void subtraction_gaussian_backward_query_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor query_diff,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels);

    void subtraction_gaussian_backward_key_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor key_diff,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels);

    void subtraction_reduce_gaussian_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels);


    void GaussianSubtraction_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output){
        shape_check(
                query,
                key);
        int batch_size = query.size(0);
        int num_head = query.size(1);
        int q_len = query.size(2);
        int k_len = key.size(3);
        int input_channels = query.size(3);
        subtraction_gaussian_forward_cuda(
                query,
                key,
                output,
                batch_size,
                num_head,
                q_len,
                k_len,
                input_channels
                );
    }


    void GaussianSubtraction_backward_query_cuda(
            const at::Tensor gradOutput,
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor gradQuery){
        shape_check(
                query,
                key);
        int batch_size = query.size(0);
        int num_head = query.size(1);
        int q_len = query.size(2);
        int k_len = key.size(3);
        int input_channels = query.size(3);
        subtraction_gaussian_backward_query_cuda(
                gradOutput,
                query,
                key,
                gradQuery,
                batch_size,
                num_head,
                q_len,
                k_len,
                input_channels
        );
    }

    void GaussianSubtraction_backward_key_cuda(
            const at::Tensor gradOutput,
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor gradKey){
        shape_check(
                query,
                key);
        int batch_size = query.size(0);
        int num_head = query.size(1);
        int q_len = query.size(2);
        int k_len = key.size(3);
//        printf("qlen: %d\n", q_len);
//        printf("klen: %d\n", k_len);
        int input_channels = query.size(3);
//        printf("query: %d, %d, %d, %d\n", query.size(0), query.size(1), query.size(2), query.size(3));
        subtraction_gaussian_backward_key_cuda(
                gradOutput,
                query,
                key,
                gradKey,
                batch_size,
                num_head,
                q_len,
                k_len,
                input_channels
        );
    }

    void GaussianSubtraction_reduce_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output){
        shape_check(
                query,
                key);
        int batch_size = query.size(0);
        int num_head = query.size(1);
        int q_len = query.size(2);
        int k_len = key.size(3);
        int input_channels = query.size(3);
        subtraction_reduce_gaussian_forward_cuda(
                query,
                key,
                output,
                batch_size,
                num_head,
                q_len,
                k_len,
                input_channels
        );
    }

}