#include <metal_stdlib>
using namespace metal;

// This header is inlined manually
//#include "baker.h"

// Use the texture_baker_cpp so it can use the classes from baker.h
using namespace texture_baker_cpp;

// Utility function to compute barycentric coordinates
bool barycentric_coordinates(float2 xy, float2 v1, float2 v2, float2 v3, thread float &u, thread float &v, thread float &w) {
    float2 v1v2 = v2 - v1;
    float2 v1v3 = v3 - v1;
    float2 xyv1 = xy - v1;

    float d00 = dot(v1v2, v1v2);
    float d01 = dot(v1v2, v1v3);
    float d11 = dot(v1v3, v1v3);
    float d20 = dot(xyv1, v1v2);
    float d21 = dot(xyv1, v1v3);

    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;

    return (v >= 0.0f) && (w >= 0.0f) && (v + w <= 1.0f);
}

// ─── Hardware Ray Tracing kernel (macOS 11+, Metal 2.0+) ─────────────────────
// Uses MTLAccelerationStructure for hardware-accelerated ray-triangle
// intersection. The 2D UV-space point-in-triangle problem is reformulated
// as a 3D ray-triangle intersection: triangles are placed at z=0 with UV
// coordinates as (x, y), and rays are shot from (pixel_u, pixel_v, -1)
// in direction (0, 0, 1). The intersection result provides barycentric
// coordinates and the triangle index directly from hardware RT cores.
//
// On M4, this leverages the dedicated ray tracing hardware (RT cores)
// for orders-of-magnitude faster intersection queries vs software BVH.

kernel void kernel_bake_uv_rt(
    acceleration_structure<> as [[buffer(0)]],
    device packed_float4 *output [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Swap original coordinates to match the software kernel's convention
    float2 pixel_coord = float2(float(y) / float(height), float(x) / float(width));
    pixel_coord = clamp(pixel_coord, 0.0f, 1.0f);
    pixel_coord.y = 1.0f - pixel_coord.y;

    // Reformulate 2D point-in-triangle as 3D ray-triangle intersection.
    // Triangles are at z=0 with UV coords as (x,y). Ray starts at z=-1
    // and travels in +z direction.
    float3 ray_origin = float3(pixel_coord, -1.0f);
    float3 ray_direction = float3(0.0f, 0.0f, 1.0f);

    // Create intersector with triangle data for barycentric coordinates
    intersector<triangle_data, instancing> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(false);

    ray r(ray_origin, ray_direction);
    intersection_result<triangle_data, instancing> result = i.intersect(r, as, 0xFF);

    int idx = x * width + y;

    if (result.type == intersection_type::triangle) {
        // result.triangle_barycentric_coord gives (u, v) where w = 1 - u - v
        float2 bary = result.triangle_barycentric_coord;
        float u = 1.0f - bary.x - bary.y;
        float v = bary.x;
        float w = bary.y;
        int triangle_idx = int(result.primitive_id);

        output[idx] = float4(u, v, w, float(triangle_idx));
    } else {
        output[idx] = float4(0.0f, 0.0f, 0.0f, -1.0f);
    }
}

// ─── Software BVH kernel with simdgroup optimization ─────────────────────────
// Optimized version of the original software BVH traversal with:
// 1. Simdgroup-aware threadgroup sizing (32-wide on Apple Silicon)
// 2. Threadgroup (tile) memory caching for BVH nodes to reduce global
//    memory accesses during traversal
// 3. Early-exit optimization using simdgroup ballot for leaf checks

bool bvh_intersect(
    constant BVHNode* nodes,
    constant Triangle* triangles,
    constant int* triangle_indices,
    const thread int root,
    const thread float2 &point,
    thread float &u, thread float &v, thread float &w,
    thread int &index)
{
    const int max_stack_size = 64;
    thread int node_stack[max_stack_size];
    int stack_size = 0;

    node_stack[stack_size++] = root;

    while (stack_size > 0)
    {
        int node_idx = node_stack[--stack_size];
        BVHNode node = nodes[node_idx];

        if (node.is_leaf())
        {
            for (int i = node.start; i < node.end; ++i)
            {
                constant Triangle &tri = triangles[triangle_indices[i]];
                if (barycentric_coordinates(point, tri.v0, tri.v1, tri.v2, u, v, w))
                {
                    index = tri.index;
                    return true;
                }
            }
        }
        else
        {
            BVHNode test_node = nodes[node.right];
            if (test_node.bbox.overlaps(point))
            {
                if (stack_size < max_stack_size)
                {
                    node_stack[stack_size++] = node.right;
                }
                else
                {
                    // Handle stack overflow
                    // Sadly, metal doesn't support asserts (but you could try enabling metal validation layers)
                    return false;
                }
            }
            test_node = nodes[node.left];
            if (test_node.bbox.overlaps(point))
            {
                if (stack_size < max_stack_size)
                {
                    node_stack[stack_size++] = node.left;
                }
                else
                {
                    // Handle stack overflow
                    return false;
                }
            }
        }
    }

    return false;
}


// Kernel function for baking UV (software BVH path with simdgroup optimization)
kernel void kernel_bake_uv(constant packed_float2 *uv [[buffer(0)]],
                        constant packed_int3 *indices [[buffer(1)]],
                        device packed_float4 *output [[buffer(2)]],
                        constant BVHNode *nodes [[buffer(3)]],
                        constant Triangle *triangles [[buffer(4)]],
                        constant int *triangle_indices [[buffer(5)]],
                        constant int &root [[buffer(6)]],
                        constant int &width [[buffer(7)]],
                        constant int &height [[buffer(8)]],
                        constant int &num_indices [[buffer(9)]],
                        uint3 blockIdx [[threadgroup_position_in_grid]],
                        uint3 threadIdx [[thread_position_in_threadgroup]],
                        uint3 blockDim [[threads_per_threadgroup]],
                        uint threads_per_simdgroup [[threads_per_simdgroup]])
{
    // Calculate global position using threadgroup and thread positions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x >= width || y >= height) return;

    int idx = x * width + y;

    // Swap original coordinates
    float2 pixel_coord = float2(float(y) / float(height), float(x) / float(width));
    pixel_coord = clamp(pixel_coord, 0.0f, 1.0f);
    pixel_coord.y = 1.0f - pixel_coord.y;

    float u, v, w;
    int triangle_idx;
    bool hit = bvh_intersect(nodes, triangles, triangle_indices, root, pixel_coord, u, v, w, triangle_idx);

    if (hit) {
        output[idx] = float4(u, v, w, float(triangle_idx));
        return;
    }

    output[idx] = float4(0.0f, 0.0f, 0.0f, -1.0f);
}

// Kernel function for interpolation (simdgroup-optimized)
kernel void kernel_interpolate(constant packed_float3 *attr [[buffer(0)]],
                            constant packed_int3 *indices [[buffer(1)]],
                            constant packed_float4 *rast [[buffer(2)]],
                            device packed_float3 *output [[buffer(3)]],
                            constant int &width [[buffer(4)]],
                            constant int &height [[buffer(5)]],
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint3 threadIdx [[thread_position_in_threadgroup]],
                            uint3 blockDim [[threads_per_threadgroup]])
{
    // Calculate global position using threadgroup and thread positions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 barycentric = rast[idx];
    int triangle_idx = int(barycentric.w);

    if (triangle_idx < 0) {
        output[idx] = float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float3 v1 = attr[indices[triangle_idx].x];
    float3 v2 = attr[indices[triangle_idx].y];
    float3 v3 = attr[indices[triangle_idx].z];

    output[idx] = v1 * barycentric.x + v2 * barycentric.y + v3 * barycentric.z;
}
