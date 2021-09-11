// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <stdio.h>
#include <float.h>
#include <math.h>

using namespace at;

// TODO make it in a common file
#define CUDA_KERNEL_LOOP_X(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)


namespace {
    const int CUDA_NUM_XTHREADS = 1024;
    const int CUDA_NUM_XYXTHREADS = 32;
    const int CUDA_NUM_XYYTHREADS = 32;
    const int kMaxGridNum = 65535;
    const int SDIM = 32;

    inline int GET_BLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XTHREADS - 1) / CUDA_NUM_XTHREADS);
    }

    inline int GET_XBLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XYXTHREADS - 1) / CUDA_NUM_XYXTHREADS);
    }

    inline int GET_YBLOCKS(const int N) {
        return std::min(kMaxGridNum, (N + CUDA_NUM_XYYTHREADS - 1) / CUDA_NUM_XYYTHREADS);
    }
}

template <typename scalar_t>
__global__ void subtraction_gaussian_forward_kernel(
        const scalar_t* query,
        const scalar_t* key,
        scalar_t* output,
        int xthreads,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels
        ) {
    CUDA_KERNEL_LOOP_X(index, xthreads) {
    //  printf("tid: %d \\n", index);
        const int b = index / num_head / q_len / k_len;
        const int h = (index / q_len / k_len) % num_head;
        const int p = (index / k_len) % q_len;
        const int q = index % k_len;
    //  printf("b: %d, h: %d, p: %d, q: %d \\n", b, h, p, q);
        if (index < batch_size * num_head * q_len * k_len){
            scalar_t sum = 0;
            for (int c = 0; c < input_channels; c++){
                int query_offset = b * (num_head * q_len * input_channels) + h * (q_len * input_channels) + p * input_channels + c;
                int key_offset = b * (num_head * k_len * input_channels) + h * (k_len * input_channels) + c * k_len + q;
                scalar_t dis = query[query_offset] - key[key_offset];
    //          printf("query off: %d, key off: %d, query data: %f, key data: %f, dis: %f \\n", query_offset, key_offset, query[query_offset], query[key_offset], dis);
                sum += dis * dis;
    //      printf("dis %f \\n", dis * dis);
            }
    //    printf("sum: %f \\n", sum);
            output[index] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void subtraction_gaussian_backward_query_kernel(
        const scalar_t* const output_diff,
        const scalar_t* const query_data,
        const scalar_t* const key_data,
        scalar_t* query_diff,
        int xthreads,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels) {
    CUDA_KERNEL_LOOP_X(index, xthreads) {
//        printf("tid: %d \n", index);
        const int b = index / num_head / q_len / input_channels;
        const int h = (index / q_len / input_channels) % num_head;
        const int q = (index / input_channels) % q_len;
        const int c = index % input_channels;
//        printf("b: %d, h: %d, q: %d, c: %d \n", b, h, q, c);
        if (index < batch_size * num_head * q_len * input_channels){
            scalar_t sum = 0;
            int query_offset = b * (num_head * q_len * input_channels) + h * (q_len * input_channels) + q * input_channels + c;
            scalar_t query = query_data[query_offset];
            for (int k = 0; k < k_len; k++){
                int output_offset = b * (num_head * k_len * q_len) + h * (k_len * q_len) + q * k_len + k;
                int key_offset = b * (num_head * k_len * input_channels) + h * (k_len * input_channels) + c * k_len + k;
                sum += 2 * output_diff[output_offset] * (query - key_data[key_offset]);
//                scalar_t output_ind= output_diff[output_offset];
//                scalar_t key_ind= key_data[key_offset];
//            printf("query off: %d, key off: %d, output off: %d, query data: %f, key data: %f, output data: %f\n", query_offset, key_offset, output_offset, query, key_ind, output_ind);
            }
//        printf("sum: %f \n", sum);
            query_diff[index] = sum;
        }
    }
}


template <typename scalar_t>
__global__ void subtraction_gaussian_backward_key_kernel(
        const scalar_t* const output_diff,
        const scalar_t* const query_data,
        const scalar_t* const key_data,
        scalar_t* key_diff,
        int xthreads,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels) {
    CUDA_KERNEL_LOOP_X(index, xthreads) {
//        printf("tid: %d \n", index);
        const int b = index / num_head / input_channels / k_len;
        const int h = (index / input_channels / k_len) % num_head;
        const int c = (index / k_len) % input_channels;
        const int k = index % k_len;
//        printf("b: %d, h: %d, c: %d, k: %d \n", b, h, c, k);
        if (index < batch_size * num_head * k_len * input_channels){
            scalar_t sum = 0;
            int key_offset = b * (num_head * k_len * input_channels) + h * (k_len * input_channels) + c * k_len + k;
            scalar_t key = key_data[key_offset];
            for (int q = 0; q < q_len; q++){
                int output_offset = b * (num_head * k_len * q_len) + h * (k_len * q_len) + q * k_len + k;
                int query_offset = b * (num_head * q_len * input_channels) + h * (q_len * input_channels) + q * input_channels + c;
                sum += 2 * output_diff[output_offset] * (query_data[query_offset] - key);
//                scalar_t output_ind= output_diff[output_offset];
//                scalar_t query_ind= query_data[query_offset];
//                printf("query off: %d, key off: %d, output off: %d, query data: %f, key data: %f, output data: %f\n", query_offset, key_offset, output_offset, query_ind, key, output_ind);
            }
//            printf("sum: %f \n", sum);
        key_diff[index] = -sum;
        }
    }
}


template <typename scalar_t>
__global__ void subtraction_reduce_gaussian_forward_kernel(
        const scalar_t* query,
        const scalar_t* key,
        scalar_t* output,
        int xthreads,
        int batch_size,
        int num_head,
        int q_len,
        int k_len,
        int input_channels
) {
    CUDA_KERNEL_LOOP_X(index, xthreads) {
        //  printf("tid: %d \\n", index);
        __shared__ scalar_t cdata[32][32];
        const int b = index / num_head / q_len / k_len;
        const int h = (index / q_len / k_len) % num_head;
        const int p = (index / k_len) % q_len;
        const int q = index % k_len;
        //  printf("b: %d, h: %d, p: %d, q: %d \\n", b, h, p, q);
        scalar_t sum = 0;
        if (index < batch_size * num_head * q_len * k_len){
            for (int c = 0; c < input_channels; c++){
                int query_offset = b * (num_head * q_len * input_channels) + h * (q_len * input_channels) + p * input_channels + c;
                int key_offset = b * (num_head * k_len * input_channels) + h * (k_len * input_channels) + c * k_len + q;
                scalar_t dis = query[query_offset] - key[key_offset];
                //          printf("query off: %d, key off: %d, query data: %f, key data: %f, dis: %f \\n", query_offset, key_offset, query[query_offset], query[key_offset], dis);
                cdata[index][c] = dis * dis;
                sum += dis * dis;
                //      printf("dis %f \\n", dis * dis);
            }
            __syncthreads();
//            printf("cdata0: %f  ", cdata[0]);
            int ytid = threadIdx.y;
//            printf("ytid %d \n", ytid);
            if (ytid < 16){
//                scalar_t *vcdata = cdata;
//                printf("32cdata0: %f, 32cdata16: %f\n", cdata[0], cdata[16]);
                __syncthreads();
                cdata[index][ytid] += cdata[index][ytid + 16];
                __syncthreads();
//                printf("16cdata0: %f\n", cdata[0]);
                __syncthreads();
                cdata[index][ytid] += cdata[index][ytid + 8];
                __syncthreads();
//                printf("8cdata0: %f\n", cdata[0]);
                cdata[index][ytid] += cdata[index][ytid + 4];
                __syncthreads();
//                printf("4cdata0: %f\n", cdata[0]);
                cdata[index][ytid] += cdata[index][ytid + 2];
                __syncthreads();
//                printf("2cdata0: %f\n", cdata[0]);
                cdata[index][ytid] += cdata[index][ytid + 1];
                __syncthreads();
//                printf("1cdata0: %f\n", cdata[0]);
            }
            if (ytid == 0) {output[index]= cdata[index][0];}

        }
    }
}


namespace SOFT {
    void subtraction_gaussian_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels) {
        const int nx = batch_size * num_head * q_len * k_len;
        at::cuda::CUDAGuard device_guard(query.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                query.scalar_type(), "subtraction_gaussian_forward_gpu", ([&] {
                const scalar_t* query_ = query.data_ptr<scalar_t>();
                const scalar_t* key_ = key.data_ptr<scalar_t>();
                scalar_t* output_ = output.data_ptr<scalar_t>();

                subtraction_gaussian_forward_kernel<<<GET_BLOCKS(nx), CUDA_NUM_XTHREADS, 0, stream>>>(
                        query_,
                        key_,
                        output_,
                        nx,
                        batch_size,
                        num_head,
                        q_len,
                        k_len,
                        input_channels);
            }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(
                    "error in subtraction_gaussian_forward_cuda: %s\n",
                    cudaGetErrorString(err));
        }
    }

    void subtraction_gaussian_backward_query_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor query_diff,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels) {
        const int nx = batch_size * num_head * q_len * input_channels;
//        printf("query nx: %d \n", nx);
        at::cuda::CUDAGuard device_guard(query_data.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                query_data.scalar_type(), "subtraction_gaussian_backward_query_gpu", ([&] {
                    const scalar_t* output_diff_ = output_diff.data_ptr<scalar_t>();
                    const scalar_t* query_data_ = query_data.data_ptr<scalar_t>();
                    const scalar_t* key_data_ = key_data.data_ptr<scalar_t>();
                    scalar_t* query_diff_ = query_diff.data_ptr<scalar_t>();

                    subtraction_gaussian_backward_query_kernel<<<GET_BLOCKS(nx), CUDA_NUM_XTHREADS, 0, stream>>>(
                            output_diff_,
                            query_data_,
                            key_data_,
                            query_diff_,
                            nx,
                            batch_size,
                            num_head,
                            q_len,
                            k_len,
                            input_channels);
                }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(
                    "error in subtraction_gaussian_backward_query_cuda: %s\n",
                    cudaGetErrorString(err));
        }
    }

    void subtraction_gaussian_backward_key_cuda(
            const at::Tensor output_diff,
            const at::Tensor query_data,
            const at::Tensor key_data,
            at::Tensor key_diff,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels) {
        const int nx = batch_size * num_head * k_len * input_channels;
//        printf("key nx: %d \n", nx);
//        printf("key len: %d \n", k_len);
//        printf("key channel: %d \n", input_channels);
        at::cuda::CUDAGuard device_guard(query_data.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                query_data.scalar_type(), "subtraction_gaussian_backward_key_gpu", ([&] {
                    const scalar_t* output_diff_ = output_diff.data_ptr<scalar_t>();
                    const scalar_t* query_data_ = query_data.data_ptr<scalar_t>();
                    const scalar_t* key_data_ = key_data.data_ptr<scalar_t>();
                    scalar_t* key_diff_ = key_diff.data_ptr<scalar_t>();

                    subtraction_gaussian_backward_key_kernel<<<GET_BLOCKS(nx), CUDA_NUM_XTHREADS, 0, stream>>>(
                            output_diff_,
                            query_data_,
                            key_data_,
                            key_diff_,
                            nx,
                            batch_size,
                            num_head,
                            q_len,
                            k_len,
                            input_channels);
                }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(
                    "error in subtraction_gaussian_backward_key_cuda: %s\n",
                    cudaGetErrorString(err));
        }
    }


    void subtraction_reduce_gaussian_forward_cuda(
            const at::Tensor query,
            const at::Tensor key,
            at::Tensor output,
            int batch_size,
            int num_head,
            int q_len,
            int k_len,
            int input_channels) {
        const int nx = batch_size * num_head * q_len * k_len;
        const int ny = input_channels;
        at::cuda::CUDAGuard device_guard(query.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                query.scalar_type(), "subtraction_reduce_gaussian_forward_gpu", ([&] {
                    const scalar_t* query_ = query.data_ptr<scalar_t>();
                    const scalar_t* key_ = key.data_ptr<scalar_t>();
                    scalar_t* output_ = output.data_ptr<scalar_t>();
                    dim3 block(CUDA_NUM_XYXTHREADS, CUDA_NUM_XYYTHREADS);
                    dim3 grid(GET_XBLOCKS(nx), GET_YBLOCKS(ny));

                    subtraction_reduce_gaussian_forward_kernel<<<grid, block, 0, stream>>>(
                            query_,
                            key_,
                            output_,
                            nx,
                            batch_size,
                            num_head,
                            q_len,
                            k_len,
                            input_channels);
                }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(
                    "error in subtraction_gaussian_forward_cuda: %s\n",
                    cudaGetErrorString(err));
        }
    }
}
