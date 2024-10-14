#pragma once

/** Computer one butterfly in forward NTT
 * x[0] = x[0] + pow * x[1] % mod
 * x[1] = x[0] - pow * x[1] % mod
 * @param[inout] x Values to operate, two int64_t, x[0] and x[1]
 * @param[in] mod The modulus
 * @param[in] pow The pre-computated one twiddle
 */
__device__ __forceinline__ void ct_butterfly(uint64_t &x, uint64_t &y,
                                             const DMulUIntMod &tw,
                                             const uint64_t &mod)
{
    uint64_t mod2 = 2 * mod;
    uint64_t tw_y = multiply_and_reduce_shoup_lazy(y, tw, mod);
    if (x >= mod2)
        x -= mod2;
    y = x + mod2 - tw_y;
    x += tw_y;
}

/** Computer one butterfly in inverse NTT
 * x[0] = (x[0] + pow * x[1]) / 2 % mod
 * x[1] = (x[0] - pow * x[1]) / 2 % mod
 * @param[inout] x Value to operate
 * @param[in] mod The modulus
 * @param[in] pow The pre-computated one twiddle
 */
__device__ __forceinline__ void gs_butterfly(uint64_t &x, uint64_t &y,
                                             const DMulUIntMod &tw,
                                             const uint64_t &mod)
{
    uint64_t mod2 = 2 * mod;
    uint64_t t = x + mod2 - y;
    uint64_t s = x + y;
    if (s >= mod2)
        s -= mod2;
    // div-2 mod
    if (t & 1)
        s += mod;
    x = s >> 1;
    y = multiply_and_reduce_shoup_lazy(t, tw, mod);
}

__device__ __forceinline__ void fntt8(uint64_t *s,
                                      const DMulUIntMod *tw,
                                      uint64_t tw_idx,
                                      uint64_t mod)
{
    // stage 1
    ct_butterfly(s[0], s[4], tw[tw_idx], mod);
    ct_butterfly(s[1], s[5], tw[tw_idx], mod);
    ct_butterfly(s[2], s[6], tw[tw_idx], mod);
    ct_butterfly(s[3], s[7], tw[tw_idx], mod);
    // stage 2
    ct_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    ct_butterfly(s[1], s[3], tw[2 * tw_idx], mod);
    ct_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    ct_butterfly(s[5], s[7], tw[2 * tw_idx + 1], mod);
    // stage 3
    ct_butterfly(s[0], s[1], tw[4 * tw_idx], mod);
    ct_butterfly(s[2], s[3], tw[4 * tw_idx + 1], mod);
    ct_butterfly(s[4], s[5], tw[4 * tw_idx + 2], mod);
    ct_butterfly(s[6], s[7], tw[4 * tw_idx + 3], mod);
}

__device__ __forceinline__ void fntt4(uint64_t *s,
                                      const DMulUIntMod *tw,
                                      uint64_t tw_idx,
                                      uint64_t mod)
{
    // stage 1
    ct_butterfly(s[0], s[2], tw[tw_idx], mod);
    ct_butterfly(s[1], s[3], tw[tw_idx], mod);
    // stage 2
    ct_butterfly(s[0], s[1], tw[2 * tw_idx], mod);
    ct_butterfly(s[2], s[3], tw[2 * tw_idx + 1], mod);
}

__device__ __forceinline__ void intt8(uint64_t *s,
                                      const DMulUIntMod *tw,
                                      uint64_t tw_idx,
                                      uint64_t mod)
{
    // stage 1
    gs_butterfly(s[0], s[1], tw[4 * tw_idx], mod);
    gs_butterfly(s[2], s[3], tw[4 * tw_idx + 1], mod);
    gs_butterfly(s[4], s[5], tw[4 * tw_idx + 2], mod);
    gs_butterfly(s[6], s[7], tw[4 * tw_idx + 3], mod);

    // stage 2
    gs_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    gs_butterfly(s[1], s[3], tw[2 * tw_idx], mod);
    gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    gs_butterfly(s[5], s[7], tw[2 * tw_idx + 1], mod);
    // stage 3
    gs_butterfly(s[0], s[4], tw[tw_idx], mod);
    gs_butterfly(s[1], s[5], tw[tw_idx], mod);
    gs_butterfly(s[2], s[6], tw[tw_idx], mod);
    gs_butterfly(s[3], s[7], tw[tw_idx], mod);
}

__device__ __forceinline__ void intt4(uint64_t *s,
                                      const DMulUIntMod *tw,
                                      uint64_t tw_idx,
                                      uint64_t mod)
{
    // stage 1
    gs_butterfly(s[0], s[2], tw[2 * tw_idx], mod);
    gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], mod);
    // stage 2
    gs_butterfly(s[0], s[4], tw[tw_idx], mod);
    gs_butterfly(s[2], s[6], tw[tw_idx], mod);
}
