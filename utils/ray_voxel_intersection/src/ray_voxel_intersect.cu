#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"  // required for float3 vector math
#define NUM_THREADS 1024

__device__ int sign(float x) { 

	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

/**
 * ray marching
 * @param batch_size number of pixels
 * @param max_n maximum possible intersections
 * @param xyzmin min volume bound
 * @param xyzmax max volume bound
 * @param voxel_num voxel grid size
 * @param voxel_size size of each voxel
 * @param o ray origins
 * @param v ray directions
 * @param intersection ray-voxel intersection buffer
 * @param intersection_num number of intersections for each ray
 * @param tns distance between ray origin and intersected location
 */
__global__ void ray_voxel_intersect_kernel(
            int batch_size, int max_n,
            const float xyzmin, const float xyzmax, 
            const float voxel_num, const float voxel_size,
            const float *__restrict__ o,
            const float *__restrict__ v,
            float *__restrict__ intersection,
            int *__restrict__ intersect_num,
            float *__restrict__ tns) {

    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if(batch_index >= batch_size){
        return;
    }

    // set up index
    o += batch_index*3;
    v += batch_index*3;
    intersection += batch_index*max_n*3;
    intersect_num += batch_index;
    tns += batch_index*max_n;

    // "normalize" the coordinate to make it appear that the size of each voxel is one
    float ox = (o[0]-xyzmin) / voxel_size;
    float oy = (o[1]-xyzmin) / voxel_size;
    float oz = (o[2]-xyzmin) / voxel_size;


    float3 dir = make_float3(v[0],v[1],v[2]);
    float3 ori = make_float3(ox,oy,oz);

    bool is_inside = (ox >= 0) & (oy >= 0) & (oz >= 0) 
                & (ox <= voxel_num) & (oy <= voxel_num) & (oz <= voxel_num);

    // if inside the volume (canvas)
    // we treat the origin as the hit entry (the first intersection)
    if (is_inside) { 
        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;
        tns[0] = 0.0f;
    } else {
        // ray bounding volume intersection
        /*
           Slap method - determine whether the ray hit the "voxel" (the whole volume here):
           Algorithm:
            - A voxel is composed by the intersection of three pairs of planes.
            - For each pairs of planes, solving 
              {t_0*d_x + o_x = 0 
               t_1*d_x + o_x = N}, where N is the size of the volume along x dimension. 
              Let t_min^(x) = min(t0, t1) and t_max^(x) = max(t_0, t_1)
            - Simarily, solving t_min^(y), t_max^(y), t_min^(z), t_max^(z)
            - Let t_min = max(t_min^(x), t_min^(y), t_min^(z))
            - Let t_max = min(t_max^(x), t_max^(y), t_max^(z))
            - If t_min > t_max, the ray misses the voxel
            - Otherwise, the rays hits the voxel and the hit entry (the first intersection)
              is o + t_min * d
        */
        float t0 = (-ori.x)/dir.x;
        float t1 = (voxel_num-ori.x)/dir.x;
        float tmin = fminf(t0,t1);
        float tmax = fmaxf(t0,t1);

        t0 = (-ori.y)/dir.y;
        t1 = (voxel_num-ori.y)/dir.y;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        t0 = (-ori.z)/dir.z;
        t1 = (voxel_num-ori.z)/dir.z;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        // Now mark the hit entry as the origin for computing the next intersection
        // clamp() makes sure the hit entry is in bound
        ori.x = clamp(ori.x+dir.x*tmin, 0.0f, voxel_num);
        ori.y = clamp(ori.y+dir.y*tmin, 0.0f, voxel_num);
        ori.z = clamp(ori.z+dir.z*tmin, 0.0f, voxel_num);

        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;

        // a miss, exit
        if (tmin > tmax) {
            return;
        } else {
            tns[0] = tmin;
        }
    }

    float t_now = tns[0];
    tns[0] *= voxel_size;
    intersect_num[0] += 1;

    intersection += 3;
    tns += 1;

    float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
    float3 bound;

    float tx;
    float ty;
    float tz;
    float tnext;


    /* 
       Now compute the other intersections iteratively
    */
    while (true) {
        // get candidate bounds (three planes) for next intersection
        // note: step is (sign(d_x), sign(d_y), sign(d_z))
        bound = floor(ori*step+1.0f)*step; 
        // the exit distance to the three surfaces
        tx = (bound.x-ori.x) / dir.x; 
        ty = (bound.y-ori.y) / dir.y;
        tz = (bound.z-ori.z) / dir.z;

        // tnext = min(tx, ty, tz) 
        // and the exit point of the voxel is o + tnext * d
        tnext = fminf(tx, fminf(ty,tz));
        // update o to the exit point
        ori += (dir*tnext);
        t_now += tnext;

        // enforce the point to be at the hitted plane (drifting error introduced) 
        // The check is needed to avoid numerical errors that may occur 
        // when calculating the intersection point.
        if (tnext == tx) {
            ori.x = bound.x;
        } else if (tnext == ty) {
            ori.y = bound.y;
        } else { 
            ori.z = bound.z;
        }

        if (ori.x < 0 | ori.y < 0 | ori.z < 0 | 
            ori.x > voxel_num | ori.y > voxel_num | ori.z > voxel_num) {
            return;
        }

        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;
        intersect_num[0] += 1;
        tns[0] = t_now*voxel_size;

        intersection += 3;
        tns += 1;
    }
}


void ray_voxel_intersect_wrapper(
  int device_id,
  int batch_size, int max_n,
  const float xyzmin, const float xyzmax, 
  const float voxel_num, const float voxel_size,
  const float *o, const float *v, 
  float *intersection, int *intersect_num, float *tns){

  cudaSetDevice(device_id);

  ray_voxel_intersect_kernel<<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS>>>(
      batch_size, max_n,
      xyzmin, xyzmax, voxel_num, voxel_size,
      o, v, 
      intersection, intersect_num, tns);
  
  CUDA_CHECK_ERRORS();
  cudaDeviceSynchronize();
}



/**
 * ray marching with occupancy mask
 * @param batch_size number of pixels
 * @param max_n maximum possible intersections
 * @param xyzmin min volume bound
 * @param xyzmax max volume bound
 * @param voxel_num voxel grid size
 * @param voxel_size size of each voxel
 * @param mask_scale relative scale of the mask in respect to the voxel grid
 * @param o ray origins
 * @param v ray directions
 * @param mask occupancy mask
 * @param intersection ray-voxel intersection buffer
 * @param intersection_num number of intersections for each ray
 * @param tns distance between ray origin and intersected location
 */
 __global__ void masked_intersect_kernel(
    int batch_size, int max_n,
    const float xyzmin, const float xyzmax, 
    const float voxel_num, const float voxel_size, const float mask_scale,
    const float *__restrict__ o,
    const float *__restrict__ v,
    const bool *__restrict__ mask,
    float *__restrict__ intersection,
    int *__restrict__ intersect_num,
    float *__restrict__ tns) {

    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 
    
    if(batch_index >= batch_size){
        return;
    }
    
    // set up index
    o += batch_index*3;
    v += batch_index*3;
    intersection += batch_index*max_n*6;
    intersect_num += batch_index;
    tns += batch_index*max_n;

    float ox = (o[0]-xyzmin) / voxel_size;
    float oy = (o[1]-xyzmin) / voxel_size;
    float oz = (o[2]-xyzmin) / voxel_size;

    
    float3 dir = make_float3(v[0],v[1],v[2]);
    float3 ori = make_float3(ox,oy,oz);
    float3 ori_last;
    float t_now;

    bool is_inside = (ox >= 0) & (oy >= 0) & (oz >= 0) 
                    & (ox <= voxel_num) & (oy <= voxel_num) & (oz <= voxel_num);

    if (is_inside) {
        ori_last = make_float3(ori.x, ori.y, ori.z);
        t_now = 0.0f;
    } else {
        // ray bounding volume intersection
        float t0 = (-ori.x)/dir.x;
        float t1 = (voxel_num-ori.x)/dir.x;
        float tmin = fminf(t0,t1);
        float tmax = fmaxf(t0,t1);

        t0 = (-ori.y)/dir.y;
        t1 = (voxel_num-ori.y)/dir.y;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        t0 = (-ori.z)/dir.z;
        t1 = (voxel_num-ori.z)/dir.z;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        ori.x = clamp(ori.x+dir.x*tmin, 0.0f, voxel_num);
        ori.y = clamp(ori.y+dir.y*tmin, 0.0f, voxel_num);
        ori.z = clamp(ori.z+dir.z*tmin, 0.0f, voxel_num);
        ori_last = make_float3(ori.x,ori.y,ori.z);
        // a miss, exit
        if (tmin > tmax) {
            return;
        } else {
            t_now = tmin;
        }
    }

    float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
    float3 bound;

    float tx;
    float ty;
    float tz;
    float tnext;
    // The number of voxels that share one mask are 1/mask_scale
    int mask_size = int(voxel_num * mask_scale);


    while (true) {
        bound = floor(ori_last*step+1.0f)*step; // get candidate bounds for next intersection
        tx = (bound.x-ori_last.x) / dir.x;
        ty = (bound.y-ori_last.y) / dir.y;
        tz = (bound.z-ori_last.z) / dir.z;

        tnext = fminf(tx, fminf(ty,tz));
        ori = ori_last + (dir*tnext);
        t_now += tnext;

        // enforce the point to be at the hitted plane (drifting error introduced) 
        if (tnext == tx) {
            ori.x = bound.x;
        } else if (tnext == ty) {
            ori.y = bound.y;
        } else { 
            ori.z = bound.z;
        }

        // check if exceed the boundary
        if (ori.x < 0 | ori.y < 0 | ori.z < 0 | 
            ori.x > voxel_num | ori.y > voxel_num | ori.z > voxel_num) {
            return;
        }

        // calc the index for mask of voxel 
        // Note that mutiple voxels can share one mask
        // 
        // The voxel index (i, j, k) = int(corner) since corner is under the voxels basis
        // Thus int(corner*mask_scale) is the mask voxel index (i, j, k)
        float3 corner = fminf(ori_last+1e-4f,ori+1e-4f)*mask_scale;
        
        int corner_index = int(corner.z)*mask_size*mask_size + int(corner.y)*mask_size + int(corner.x);
        
        // only mark as intersected when the voxel is not masked
        // (and intersection must be marked pairwise)
        if (mask[corner_index]) {
            intersection[0] = ori_last.x;
            intersection[1] = ori_last.y;
            intersection[2] = ori_last.z;
            intersection[3] = ori.x;
            intersection[4] = ori.y;
            intersection[5] = ori.z;
            intersect_num[0] += 1;
            tns[0] = t_now*voxel_size; // map back the normalized depth t

            intersection += 6;
            tns += 1;
        }

        ori_last.x = ori.x;
        ori_last.y = ori.y;
        ori_last.z = ori.z;
    }
}



void masked_intersect_wrapper(
    int device_id,
    int batch_size, int max_n,
    const float xyzmin, const float xyzmax, 
    const float voxel_num, const float voxel_size, const float mask_scale,
    const float *o, const float *v, const bool *mask,
    float *intersection, int *intersect_num, float *tns){

    cudaSetDevice(device_id);

    masked_intersect_kernel<<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS>>>(
    batch_size, max_n,
    xyzmin, xyzmax, voxel_num, voxel_size, mask_scale,
    o, v, mask, 
    intersection, intersect_num, tns);

    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}
