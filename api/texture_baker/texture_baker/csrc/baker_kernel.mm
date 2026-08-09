#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include "baker.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <filesystem>
#include <algorithm>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper function to create a compute pipeline state object (PSO).
static inline id<MTLComputePipelineState> createComputePipelineState(id<MTLDevice> device, NSString* fullSource, std::string kernel_name) {
    NSError *error = nil;

    // Load the custom kernel shader.
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    // Add the preprocessor macro "__METAL__"
    options.preprocessorMacros = @{@"__METAL__": @""};
    // Set language version to 3.0 for hardware RT and simdgroup support
    if (@available(macOS 13.0, *)) {
        options.languageVersion = MTLLanguageVersion3_0;
    } else if (@available(macOS 11.0, *)) {
        options.languageVersion = MTLLanguageVersion2_3;
    }
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource: fullSource options:options error:&error];
    TORCH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    id<MTLFunction> customKernelFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(customKernelFunction, "Failed to create function state object for ", kernel_name.c_str());

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:customKernelFunction error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    return pso;
}

// Check if hardware ray tracing is supported on this device.
// Metal ray tracing requires macOS 11.0+ and Apple Silicon (Apple7 GPU family or later).
static inline bool isHardwareRTSupported(id<MTLDevice> device) {
    if (@available(macOS 11.0, *)) {
        // Check if the device supports ray tracing (Apple7 = M1+)
        return [device supportsFamily:MTLGPUFamilyApple7];
    }
    return false;
}

// Build a Metal acceleration structure from UV triangle data for hardware RT.
// The 2D UV coordinates are embedded as the x,y components of 3D vertices at z=0.
static inline id<MTLAccelerationStructure> buildAccelerationStructure(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const tb_float2 *vertices,
    const tb_int3 *indices,
    int64_t num_indices)
{
    if (@available(macOS 11.0, *)) {
        // Create 3D vertex data from 2D UV coordinates (z=0)
        // Each UV vertex becomes a float3 (uv.x, uv.y, 0.0)
        // We need to expand the indexed vertices to a flat vertex buffer
        // since MTLAccelerationStructureTriangleGeometryDescriptor expects
        // a vertex buffer with vertexStride.
        int64_t num_vertices = 0;
        for (int64_t i = 0; i < num_indices; ++i) {
            num_vertices = std::max(num_vertices, (int64_t)std::max({indices[i].x, indices[i].y, indices[i].z}) + 1);
        }

        // Create 3D vertex buffer (uv.x, uv.y, 0.0) for each vertex
        size_t vertexBufferSize = num_vertices * sizeof(float) * 3;
        id<MTLBuffer> vertexBuffer = [device newBufferWithLength:vertexBufferSize
                                                         options:MTLResourceStorageModeShared];
        float3 *vertPtr = (float3 *)vertexBuffer.contents;
        for (int64_t i = 0; i < num_vertices; ++i) {
            vertPtr[i] = float3(vertices[i].x, vertices[i].y, 0.0f);
        }

        // Create index buffer (uint32)
        size_t indexBufferSize = num_indices * 3 * sizeof(uint32_t);
        id<MTLBuffer> indexBuffer = [device newBufferWithLength:indexBufferSize
                                                        options:MTLResourceStorageModeShared];
        uint32_t *idxPtr = (uint32_t *)indexBuffer.contents;
        for (int64_t i = 0; i < num_indices; ++i) {
            idxPtr[i * 3 + 0] = (uint32_t)indices[i].x;
            idxPtr[i * 3 + 1] = (uint32_t)indices[i].y;
            idxPtr[i * 3 + 2] = (uint32_t)indices[i].z;
        }

        // Create triangle geometry descriptor
        MTLAccelerationStructureTriangleGeometryDescriptor *geometryDescriptor =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geometryDescriptor.vertexBuffer = vertexBuffer;
        geometryDescriptor.vertexBufferOffset = 0;
        geometryDescriptor.vertexStride = sizeof(float3);
        geometryDescriptor.vertexFormat = MTLAttributeFormatFloat3;
        geometryDescriptor.indexBuffer = indexBuffer;
        geometryDescriptor.indexBufferOffset = 0;
        geometryDescriptor.indexType = MTLIndexTypeUInt32;
        geometryDescriptor.triangleCount = (NSUInteger)num_indices;

        // Create primitive acceleration structure descriptor
        MTLPrimitiveAccelerationStructureDescriptor *primitiveDescriptor =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        primitiveDescriptor.geometryDescriptors = @[ geometryDescriptor ];
        primitiveDescriptor.usage = MTLAccelerationStructureUsageRefit;

        // Query sizes
        MTLAccelerationStructureSizes sizes =
            [device accelerationStructureSizesWithDescriptor:primitiveDescriptor];

        // Allocate acceleration structure
        id<MTLAccelerationStructure> accelerationStructure =
            [device makeAccelerationStructureWithSize:sizes.accelerationStructureSize];

        // Allocate scratch buffer
        id<MTLBuffer> scratchBuffer =
            [device newBufferWithLength:sizes.buildScratchBufferSize
                                options:MTLResourceStorageModePrivate];

        // Encode build
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> commandEncoder =
            [commandBuffer makeAccelerationStructureCommandEncoder];

        [commandEncoder buildAccelerationStructure:accelerationStructure
                                        descriptor:primitiveDescriptor
                                     scratchBuffer:scratchBuffer
                               scratchBufferOffset:0];

        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        return accelerationStructure;
    }
    return nil;
}

std::filesystem::path get_extension_path() {
    // Ensure the GIL is held before calling any Python C API function
    PyGILState_STATE gstate = PyGILState_Ensure();

    const char* module_name = "texture_baker";

    // Import the module by name
    PyObject* module = PyImport_ImportModule(module_name);
    if (!module) {
        PyGILState_Release(gstate);
        throw std::runtime_error("Could not import the module: " + std::string(module_name));
    }

    // Get the filename of the module
    PyObject* filename_obj = PyModule_GetFilenameObject(module);
    if (filename_obj) {
        std::string path = PyUnicode_AsUTF8(filename_obj);
        Py_DECREF(filename_obj);
        PyGILState_Release(gstate);

        // Get the directory part of the path (removing the __init__.py)
        std::filesystem::path module_path = std::filesystem::path(path).parent_path();

        // Append the 'csrc' directory to the path
        module_path /= "csrc";

        return module_path;
    } else {
        PyGILState_Release(gstate);
        throw std::runtime_error("Could not retrieve the module filename.");
    }
}

NSString *get_shader_sources_as_string()
{
    const std::filesystem::path csrc_path = get_extension_path();
    const std::string shader_path = (csrc_path / "baker_kernel.metal").string();
    const std::string shader_header_path = (csrc_path / "baker.h").string();
    // Load the Metal shader from the specified path
    NSError *error = nil;

    NSString* shaderHeaderSource = [
        NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader_header_path.c_str()]
        encoding:NSUTF8StringEncoding
        error:&error];
    if (error) {
        throw std::runtime_error("Failed to load baker.h: " + std::string(error.localizedDescription.UTF8String));
    }

    NSString* shaderSource = [
        NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader_path.c_str()]
        encoding:NSUTF8StringEncoding
        error:&error];
    if (error) {
        throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }

    NSString *fullSource = [shaderHeaderSource stringByAppendingString:shaderSource];

    return fullSource;
}

namespace texture_baker_cpp
{
    torch::Tensor rasterize_gpu(
        torch::Tensor uv,
        torch::Tensor indices,
        int64_t bake_resolution)
    {
        TORCH_CHECK(uv.device().is_mps(), "uv must be a MPS tensor");
        TORCH_CHECK(uv.is_contiguous(), "uv must be contiguous");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

        TORCH_CHECK(uv.scalar_type() == torch::kFloat32, "Unsupported data type: ", indices.scalar_type());
        TORCH_CHECK(indices.scalar_type() == torch::kInt32, "Unsupported data type: ", indices.scalar_type());

        torch::Tensor rast_result = torch::empty({bake_resolution, bake_resolution, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)).contiguous();

        @autoreleasepool {
            auto vertices_cpu = uv.contiguous().cpu();
            auto indices_cpu = indices.contiguous().cpu();

            const tb_float2 *vertices_cpu_ptr = (tb_float2*)vertices_cpu.contiguous().data_ptr<float>();
            const tb_int3 *tris_cpu_ptr = (tb_int3*)indices_cpu.contiguous().data_ptr<int>();

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            NSString *fullSource = get_shader_sources_as_string();

            // Check for hardware ray tracing support
            bool useHardwareRT = isHardwareRTSupported(device);

            id<MTLAccelerationStructure> accelerationStructure = nil;
            id<MTLComputePipelineState> bake_uv_PSO = nil;

            if (useHardwareRT) {
                // Build acceleration structure and create RT kernel PSO
                id<MTLCommandQueue> commandQueue = device.newCommandQueue;
                accelerationStructure = buildAccelerationStructure(
                    device, commandQueue, vertices_cpu_ptr, tris_cpu_ptr, indices.size(0));

                if (accelerationStructure) {
                    bake_uv_PSO = createComputePipelineState(device, fullSource, "kernel_bake_uv_rt");
                } else {
                    useHardwareRT = false;
                }
            }

            // Fallback to software BVH kernel
            if (!useHardwareRT) {
                BVH bvh;
                bvh.build(vertices_cpu_ptr, tris_cpu_ptr, indices.size(0));
                bake_uv_PSO = createComputePipelineState(device, fullSource, "kernel_bake_uv");
            }

            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
                // Start a compute pass.
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

                // Get Metal buffers directly from PyTorch tensors
                auto uv_buf = getMTLBufferStorage(uv.contiguous());
                auto indices_buf = getMTLBufferStorage(indices.contiguous());
                auto rast_result_buf = getMTLBufferStorage(rast_result);

                const int width = bake_resolution;
                const int height = bake_resolution;
                const int num_indices = indices.size(0);

                if (useHardwareRT && accelerationStructure) {
                    // ─── Hardware RT path ───────────────────────────────────────
                    // Bind acceleration structure at buffer index 0
                    [computeEncoder setAccelerationStructure:accelerationStructure atBufferIndex:0];
                    [computeEncoder setBuffer:rast_result_buf offset:rast_result.storage_offset() * rast_result.element_size() atIndex:1];
                    [computeEncoder setBytes:&width length:sizeof(int) atIndex:2];
                    [computeEncoder setBytes:&height length:sizeof(int) atIndex:3];

                    // Simdgroup-aware threadgroup sizing: use threadExecutionWidth
                    // from the PSO to align threadgroups to the simdgroup width
                    // (32 on Apple Silicon). This ensures full simdgroup utilization.
                    NSUInteger simdWidth = bake_uv_PSO.threadExecutionWidth;
                    NSUInteger maxThreadsPerGroup = bake_uv_PSO.maxTotalThreadsPerThreadgroup;

                    // Use a 2D threadgroup that's a multiple of the simdgroup width.
                    // For 32-wide simdgroups: 32x4 = 128 threads per threadgroup.
                    NSUInteger blockX = simdWidth;  // 32 on Apple Silicon
                    NSUInteger blockY = std::min(maxThreadsPerGroup / simdWidth, (NSUInteger)4);
                    MTLSize threadgroupSize = MTLSizeMake(blockX, blockY, 1);
                    MTLSize numThreadgroups = MTLSizeMake(
                        (bake_resolution + blockX - 1) / blockX,
                        (bake_resolution + blockY - 1) / blockY,
                        1);

                    [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
                } else {
                    // ─── Software BVH path (with simdgroup-aware sizing) ────────
                    BVH bvh;
                    bvh.build(vertices_cpu_ptr, tris_cpu_ptr, indices.size(0));

                    const int bvh_root = bvh.root;

                    // Wrap the existing CPU memory in Metal buffers with shared memory
                    id<MTLBuffer> nodesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.nodes.data() length:sizeof(BVHNode) * bvh.nodes.size() options:MTLResourceStorageModeShared deallocator:nil];
                    id<MTLBuffer> trianglesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.triangles.data() length:sizeof(Triangle) * bvh.triangles.size() options:MTLResourceStorageModeShared deallocator:nil];
                    id<MTLBuffer> triangleIndicesBuffer = [device newBufferWithBytesNoCopy:(void*)bvh.triangle_indices.data() length:sizeof(int) * bvh.triangle_indices.size() options:MTLResourceStorageModeShared deallocator:nil];

                    [computeEncoder setBuffer:uv_buf offset:uv.storage_offset() * uv.element_size() atIndex:0];
                    [computeEncoder setBuffer:indices_buf offset:indices.storage_offset() * indices.element_size() atIndex:1];
                    [computeEncoder setBuffer:rast_result_buf offset:rast_result.storage_offset() * rast_result.element_size() atIndex:2];
                    [computeEncoder setBuffer:nodesBuffer offset:0 atIndex:3];
                    [computeEncoder setBuffer:trianglesBuffer offset:0 atIndex:4];
                    [computeEncoder setBuffer:triangleIndicesBuffer offset:0 atIndex:5];
                    [computeEncoder setBytes:&bvh_root length:sizeof(int) atIndex:6];
                    [computeEncoder setBytes:&width length:sizeof(int) atIndex:7];
                    [computeEncoder setBytes:&height length:sizeof(int) atIndex:8];
                    [computeEncoder setBytes:&num_indices length:sizeof(int) atIndex:9];

                    // Simdgroup-aware threadgroup sizing: query the actual simdgroup
                    // width from the PSO (32 on all Apple Silicon generations).
                    // Use a threadgroup that's a multiple of the simdgroup width
                    // to ensure full hardware utilization without wasted lanes.
                    NSUInteger simdWidth = bake_uv_PSO.threadExecutionWidth;
                    NSUInteger maxThreadsPerGroup = bake_uv_PSO.maxTotalThreadsPerThreadgroup;

                    // For the 2D bake grid, use simdWidth x (maxThreads/simdWidth) threadgroups.
                    // On Apple Silicon with 32-wide simdgroups and 1024 max threads:
                    // 32 x 32 = 1024 (but we cap at 16x16=256 for good occupancy).
                    NSUInteger blockX = std::min(simdWidth, (NSUInteger)32);
                    NSUInteger blockY = std::min(maxThreadsPerGroup / blockX, (NSUInteger)16);
                    // Ensure blockX * blockY doesn't exceed maxThreadsPerGroup
                    if (blockX * blockY > maxThreadsPerGroup) {
                        blockY = maxThreadsPerGroup / blockX;
                    }
                    MTLSize threadgroupSize = MTLSizeMake(blockX, blockY, 1);
                    MTLSize numThreadgroups = MTLSizeMake(
                        (bake_resolution + blockX - 1) / blockX,
                        (bake_resolution + blockY - 1) / blockY,
                        1);

                    [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
                }

                [computeEncoder endEncoding];

                // Commit the work.
                torch::mps::commit();
            });
        }

        return rast_result;
    }

    torch::Tensor interpolate_gpu(
        torch::Tensor attr,
        torch::Tensor indices,
        torch::Tensor rast)
    {
        TORCH_CHECK(attr.is_contiguous(), "attr must be contiguous");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
        TORCH_CHECK(rast.is_contiguous(), "rast must be contiguous");

        torch::Tensor pos_bake = torch::empty({rast.size(0), rast.size(1), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)).contiguous();
        std::filesystem::path csrc_path = get_extension_path();

        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            NSString *fullSource = get_shader_sources_as_string();
            // Create a compute pipeline state object using the helper function
            id<MTLComputePipelineState> interpolate_PSO = createComputePipelineState(device, fullSource, "kernel_interpolate");

            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
                // Start a compute pass.
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

                // Get Metal buffers directly from PyTorch tensors
                auto attr_buf = getMTLBufferStorage(attr.contiguous());
                auto indices_buf = getMTLBufferStorage(indices.contiguous());
                auto rast_buf = getMTLBufferStorage(rast.contiguous());
                auto pos_bake_buf = getMTLBufferStorage(pos_bake);

                int width = rast.size(0);
                int height = rast.size(1);

                [computeEncoder setComputePipelineState:interpolate_PSO];
                [computeEncoder setBuffer:attr_buf offset:attr.storage_offset() * attr.element_size() atIndex:0];
                [computeEncoder setBuffer:indices_buf offset:indices.storage_offset() * indices.element_size() atIndex:1];
                [computeEncoder setBuffer:rast_buf offset:rast.storage_offset() * rast.element_size() atIndex:2];
                [computeEncoder setBuffer:pos_bake_buf offset:pos_bake.storage_offset() * pos_bake.element_size() atIndex:3];
                [computeEncoder setBytes:&width length:sizeof(int) atIndex:4];
                [computeEncoder setBytes:&height length:sizeof(int) atIndex:5];

                // Simdgroup-aware threadgroup sizing for interpolation kernel.
                // Query threadExecutionWidth from the PSO to align to the
                // simdgroup width (32 on Apple Silicon) for full utilization.
                NSUInteger simdWidth = interpolate_PSO.threadExecutionWidth;
                NSUInteger maxThreads = interpolate_PSO.maxTotalThreadsPerThreadgroup;

                NSUInteger blockX = std::min(simdWidth, (NSUInteger)32);
                NSUInteger blockY = std::min(maxThreads / blockX, (NSUInteger)16);
                if (blockX * blockY > maxThreads) {
                    blockY = maxThreads / blockX;
                }
                MTLSize threadgroupSize = MTLSizeMake(blockX, blockY, 1);
                MTLSize numThreadgroups = MTLSizeMake(
                    (rast.size(0) + blockX - 1) / blockX,
                    (rast.size(1) + blockY - 1) / blockY,
                    1);

                [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];

                [computeEncoder endEncoding];

                // Commit the work.
                torch::mps::commit();
            });
        }

        return pos_bake;
    }

    // Registers MPS implementations
    TORCH_LIBRARY_IMPL(texture_baker_cpp, MPS, m)
    {
        m.impl("rasterize", &rasterize_gpu);
        m.impl("interpolate", &interpolate_gpu);
    }

}
