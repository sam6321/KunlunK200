// Compact KL1 CDNN helpers, copied from XDNN 2.0 gemm_libs_impl.h / findmax.h
// so we own a launchable reference without pulling the whole library TU.
// SRAM banks are separate address spaces; pointers are offsets from 0.
//
// FOOTGUN: xfence_unlock(dev) in sdcdnn_header.h expands to two statements
// ( __builtin_xpu_xfence(); __builtin_xpu_unlock(dev); ) with no do{}while(0)
// wrapper. Never use it as the body of a brace-less `if`/`for`: the unlock
// leaks out and runs unconditionally, unlocking an unlocked/invalid engine
// (mydev==-1 on non-engine cores) which 714s the card. Always brace it.
#pragma once
#include "xpu/kernel/sdcdnn_header.h"
#include "xpu/kernel/math.h"

static inline __device__ void cdnn_only_core0() {
    if (core_id() != 0) {
        return;
    }
}

// Device ALU is 32-bit. m * packed_K overflows signed int past 2 GB.
// long long is emulated; highptr already reads the top half of a global pointer.
static inline __device__ _global_ptr_ const int8_t* gptr_i8(
    _global_ptr_ const int8_t* p, long long byte_off) {
    return (_global_ptr_ const int8_t*)((long long)p + byte_off);
}

static inline __device__ _global_ptr_ float* gptr_f32(_global_ptr_ float* p,
                                                     long long elem_off) {
    return (_global_ptr_ float*)((long long)p + elem_off * (int)sizeof(float));
}

static inline __device__ _global_ptr_ const float* gptr_f32c(
    _global_ptr_ const float* p, long long elem_off) {
    return (_global_ptr_ const float*)((long long)p + elem_off * (int)sizeof(float));
}

static inline __device__ _global_ptr_ const int16_t* gptr_i16(
    _global_ptr_ const int16_t* p, long long elem_off) {
    return (_global_ptr_ const int16_t*)((long long)p + elem_off * (int)sizeof(int16_t));
}

static inline __device__ void dma_in_1d(void* l2, _global_ptr_ const void* gm,
                                        int nbytes, int dst_pos, int dtype) {
    xfence_lock(DMAIN_0);
    dma_cfg((dst_pos | dtype), (DMA_GM | dtype | highptr(gm)), 0.0f);
    dma_i(l2, lowptr(gm), nbytes);
    xfence_unlock(DMAIN_0);
}

static inline __device__ void dma_out_l2r_1d(_global_ptr_ void* gm, void* l2r,
                                             int nbytes, int dtype) {
    xfence_lock(DMAOUT);
    dma_cfg((DMA_GM | dtype | highptr(gm)), (DMA_L2R | DMA_FP32), 0.0f);
    dma_o(lowptr(gm), l2r, nbytes);
    xfence_unlock(DMAOUT);
}

static inline __device__ void dma_out_l2r_2d(_global_ptr_ void* gm, void* l2r,
                                             int rows, int cols, int dst_ld,
                                             int src_ld, int elem) {
    xfence_lock(DMAOUT);
    dma_cfg((DMA_GM | DMA_FP32 | highptr(gm)), (DMA_L2R | DMA_FP32), 0.0f);
    dma_cfg_2d(rows, dst_ld * elem, src_ld * elem);
    dma_o_2d(lowptr(gm), l2r, cols * elem);
    xfence_unlock(DMAOUT);
}

// HBM [mm, nn] int16 -> L1 via L2. Matches shuffle_l2_to_l1_int16
// with nrows=nn, ncols=mm (gemm_libs dma_shuffle_hbm_to_l1_int16).
static inline __device__ void load_int16_to_l1(_global_ptr_ const int16_t* gm,
                                               int mm, int nn, int ldn,
                                               v16i16* l1, int dst_sram) {
    int16_t* l2 = (int16_t*)0;
    int nbytes = mm * nn * (int)sizeof(int16_t);
    dma_in_1d(l2, gm, nbytes, DMA_L2_0, DMA_INT16);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT16);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int16_t* l2_ptr = l2;
    int row_stride = roundup16(nn);
    for (int i = 0; i < mm; i += NBANKS) {
        ds_shuffle_batch(&l1[l1_iter][0], l2_ptr, nn, dst_sram);
        l2_ptr += NBANKS * nn;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
    (void)ldn;
}

// HBM [mm, nn] int8 -> L1. ncols stepped by 32.
static inline __device__ void load_int8_to_l1(_global_ptr_ const int8_t* gm,
                                              int mm, int nn, v32i8* l1,
                                              int dst_sram) {
    int8_t* l2 = (int8_t*)0;
    dma_in_1d(l2, gm, mm * nn, DMA_L2_0, DMA_INT8);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(nn);
    for (int i = 0; i < mm; i += 2 * NBANKS) {
        ds_shuffle_batch(&l1[l1_iter][0], l2_ptr, nn, dst_sram);
        l2_ptr += NBANKS * nn;
        ds_shuffle_batch(&l1[l1_iter][1], l2_ptr, nn, dst_sram);
        l2_ptr += NBANKS * nn;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

// HBM [mm, nn] int16 -> L1W (or L1D with TRANS). L2.shape == [nrows, ncols].
static inline __device__ void load_int16_to_l1_coa(_global_ptr_ const int16_t* gm,
                                                   int nrows, int ncols,
                                                   v16i16* l1, int dst_sram) {
    int16_t* l2 = (int16_t*)0;
    dma_in_1d(l2, gm, nrows * ncols * (int)sizeof(int16_t), DMA_L2_0, DMA_INT16);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT16);
    ds_shuffle_cfg_blkx(ncols);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int16_t* l2_ptr = l2;
    int row_stride = roundup16(nrows);
    for (int i = 0; i < ncols; i += NBANKS) {
        ds_shuffle_coa(nrows, &l1[l1_iter][0], l2_ptr, dst_sram);
        l2_ptr += NBANKS;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

// HBM [nrows, ncols] int8 -> L1W (or L1D with TRANS).
static inline __device__ void load_int8_to_l1_coa(_global_ptr_ const int8_t* gm,
                                                  int nrows, int ncols,
                                                  v32i8* l1, int dst_sram) {
    int8_t* l2 = (int8_t*)0;
    dma_in_1d(l2, gm, nrows * ncols, DMA_L2_0, DMA_INT8);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(ncols);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(nrows);
    for (int i = 0; i < ncols; i += 2 * NBANKS) {
        ds_shuffle_coa(nrows, &l1[l1_iter][0], l2_ptr, dst_sram);
        l2_ptr += NBANKS;
        ds_shuffle_coa(nrows, &l1[l1_iter][1], l2_ptr, dst_sram);
        l2_ptr += NBANKS;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

// Identity L1E load (XDNN load_matrix_l1e_fp32_notrans). shuffle_batch here is the
// transpose path and will swap mm/nn relative to rs_row store.
static inline __device__ void load_fp32_to_l1e(_global_ptr_ const float* gm,
                                               int mm, int nn) {
    float* l2 = (float*)0;
    dma_in_1d(l2, gm, mm * nn * (int)sizeof(float), DMA_L2_0, DMA_FP32);
    v16f32* l1e = (v16f32*)0;
    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_FP32);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    float* l2_ptr = l2;
    for (int i = 0; i < nn; i += NBANKS) {
        ds_shuffle_coa(mm, &l1e[l1_iter][0], l2_ptr, DS_L1E);
        l2_ptr += NBANKS;
        l1_iter += mm;
    }
    xfence_unlock(DS_0);
}

// Batch shuffle into L1E (transpose vs coa). Use when the stream axis should be HBM W.
static inline __device__ void load_fp32_to_l1e_batch(_global_ptr_ const float* gm,
                                                     int mm, int nn) {
    float* l2 = (float*)0;
    dma_in_1d(l2, gm, mm * nn * (int)sizeof(float), DMA_L2_0, DMA_FP32);
    v16f32* l1e = (v16f32*)0;
    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_FP32);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    float* l2_ptr = l2;
    int row_stride = roundup16(nn);
    for (int i = 0; i < mm; i += NBANKS) {
        ds_shuffle_batch(&l1e[l1_iter][0], l2_ptr, nn, DS_L1E);
        l2_ptr += NBANKS * nn;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

static inline __device__ void store_l1e_fp32(_global_ptr_ float* gm, int mm, int nn,
                                             int act = EW_NOACT) {
    v16f32* l1e = (v16f32*)0;
    v16f32* l2e = (v16f32*)0;
    v16f32* l2r = (v16f32*)0;
    int nblocks = roundup_div16(nn);

    xfence_lock(EW);
    ew_cfg_activation_type(act);
    ew_cfg_coeff_scalar(1.0f, 0.0f, 0.0f);
    ew_cfg_findmax(0, 0.0f);
    ew_cfg_stream_size(mm * nblocks);
    ew_cfg_col_size(NBANKS);
    ew_dsmadd(l2e, l1e, l1e, EW_L2E);
    xfence_unlock(EW);

    xfence_lock(RS);
    rs_row_cfg_stride(1, nblocks);
    rs_row_cfg_loop(mm);
    for (int i = 0; i < nblocks; i++) {
        rs_row_batch(&l2r[i][0], &l2e[i * mm][0], NBANKS);
    }
    xfence_unlock(RS);

    dma_out_l2r_2d(gm, l2r, mm, nn, nn, roundup16(nn), (int)sizeof(float));
}

static inline __device__ void mac_int16_tile(int k, int mm, int nn) {
    v16i16* l1d = (v16i16*)0;
    v16i16* l1w = (v16i16*)0;
    v16f32* l1e = (v16f32*)0;
    int roundup_k = roundup16(k);
    int ntiles = roundup_div16(nn);
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.0f);
    mm_cfg_basic_int16(k, mm);
    for (int n_iter = 0; n_iter < ntiles; n_iter++) {
        mm_int16(&l1e[n_iter * mm], l1d, &l1w[n_iter * roundup_k]);
    }
    xfence_unlock(MAC);
}

static inline __device__ void mac_int8_tile(int k, int mm, int nn) {
    v32i8* l1d = (v32i8*)0;
    v32i8* l1w = (v32i8*)0;
    v16f32* l1e = (v16f32*)0;
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.0f);
    mm_cfg_basic_int8(k, mm, mm);
    mm_int8(l1e, l1d, l1w);
    xfence_unlock(MAC);
    (void)nn;
}

template <int MODE>
static inline __device__ void mac_int4_tile(int k, int mm, int nn) {
    v32i8* l1d = (v32i8*)0;
    v32i8* l1w = (v32i8*)0;
    v16f32* l1e = (v16f32*)0;
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.0f);
    mm_cfg_basic_int8(k, mm, mm);
    mm_int4(l1e, l1d, l1w, MODE);
    xfence_unlock(MAC);
    (void)nn;
}

static inline __device__ void dma_in_1d_dev(void* l2, _global_ptr_ const void* gm,
                                            int nbytes, int dst_pos, int dtype,
                                            int dmain) {
    xfence_lock(dmain);
    dma_cfg((dst_pos | dtype), (DMA_GM | dtype | highptr(gm)), 0.0f);
    dma_i(l2, lowptr(gm), nbytes);
    xfence_unlock(dmain);
}

static inline __device__ void dma_in_2d(void* l2, _global_ptr_ const void* gm,
                                        int rows, int cols, int src_ld, int elem,
                                        int dst_pos, int dtype) {
    xfence_lock(DMAIN_0);
    dma_cfg((dst_pos | dtype), (DMA_GM | dtype | highptr(gm)), 0.0f);
    dma_cfg_2d(rows, cols * elem, src_ld * elem);
    dma_i_2d(l2, lowptr(gm), cols * elem);
    xfence_unlock(DMAIN_0);
}

static inline __device__ void shuffle_int8_l2_to_l1d(int mm, int packed_k, v32i8* l1) {
    int8_t* l2 = (int8_t*)0;
    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(packed_k);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(packed_k);
    for (int i = 0; i < mm; i += 2 * NBANKS) {
        ds_shuffle_batch(&l1[l1_iter][0], l2_ptr, packed_k, DS_L1D);
        l2_ptr += NBANKS * packed_k;
        ds_shuffle_batch(&l1[l1_iter][1], l2_ptr, packed_k, DS_L1D);
        l2_ptr += NBANKS * packed_k;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

static inline __device__ void shuffle_int8_l2_to_l1w(int packed_k, int nn, v32i8* l1) {
    int8_t* l2 = (int8_t*)0;
    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(packed_k);
    for (int i = 0; i < nn; i += 2 * NBANKS) {
        ds_shuffle_coa(packed_k, &l1[l1_iter][0], l2_ptr, DS_L1W);
        l2_ptr += NBANKS;
        ds_shuffle_coa(packed_k, &l1[l1_iter][1], l2_ptr, DS_L1W);
        l2_ptr += NBANKS;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

// A panel [mm, packed_k] with leading dim a_ld (packed K of the full matrix).
static inline __device__ void load_int8_a_panel(_global_ptr_ const int8_t* gm,
                                                int mm, int packed_k, int a_ld,
                                                v32i8* l1) {
    int8_t* l2 = (int8_t*)0;
    if (a_ld == packed_k) {
        dma_in_1d(l2, gm, mm * packed_k, DMA_L2_0, DMA_INT8);
    } else {
        dma_in_2d(l2, gm, mm, packed_k, a_ld, 1, DMA_L2_0, DMA_INT8);
    }
    shuffle_int8_l2_to_l1d(mm, packed_k, l1);
}

// B panel [packed_k, nn] with leading dim b_ld (N of the full matrix).
static inline __device__ void load_int8_b_panel(_global_ptr_ const int8_t* gm,
                                                int packed_k, int nn, int b_ld,
                                                v32i8* l1) {
    int8_t* l2 = (int8_t*)0;
    if (b_ld == nn) {
        dma_in_1d(l2, gm, packed_k * nn, DMA_L2_0, DMA_INT8);
    } else {
        dma_in_2d(l2, gm, packed_k, nn, b_ld, 1, DMA_L2_0, DMA_INT8);
    }
    shuffle_int8_l2_to_l1w(packed_k, nn, l1);
}

// Odd cores: DMAIN_1 / DMA_L2_1 / DS_1. Even: DMAIN_0 / L2_0 / DS_0.
static inline __device__ void dma_in_int8_dev(_global_ptr_ const int8_t* gm,
                                              int rows, int cols, int src_ld,
                                              int dmain, int dst_pos) {
    int8_t* l2 = (int8_t*)0;
    xfence_lock(dmain);
    dma_cfg((dst_pos | DMA_INT8), (DMA_GM | DMA_INT8 | highptr(gm)), 0.0f);
    if (src_ld == cols) {
        dma_i(l2, lowptr(gm), rows * cols);
    } else {
        dma_cfg_2d(rows, cols, src_ld);
        dma_i_2d(l2, lowptr(gm), cols);
    }
    xfence_unlock(dmain);
}

static inline __device__ void shuffle_int8_l2_to_l1d_ds(int mm, int packed_k,
                                                        v32i8* l1, int ds) {
    int8_t* l2 = (int8_t*)0;
    xfence_lock(ds);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(packed_k);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(packed_k);
    for (int i = 0; i < mm; i += 2 * NBANKS) {
        ds_shuffle_batch(&l1[l1_iter][0], l2_ptr, packed_k, DS_L1D);
        l2_ptr += NBANKS * packed_k;
        ds_shuffle_batch(&l1[l1_iter][1], l2_ptr, packed_k, DS_L1D);
        l2_ptr += NBANKS * packed_k;
        l1_iter += row_stride;
    }
    xfence_unlock(ds);
}

static inline __device__ void shuffle_int8_l2_to_l1w_ds(int packed_k, int nn,
                                                        v32i8* l1, int ds) {
    int8_t* l2 = (int8_t*)0;
    xfence_lock(ds);
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int8_t* l2_ptr = l2;
    int row_stride = roundup16(packed_k);
    for (int i = 0; i < nn; i += 2 * NBANKS) {
        ds_shuffle_coa(packed_k, &l1[l1_iter][0], l2_ptr, DS_L1W);
        l2_ptr += NBANKS;
        ds_shuffle_coa(packed_k, &l1[l1_iter][1], l2_ptr, DS_L1W);
        l2_ptr += NBANKS;
        l1_iter += row_stride;
    }
    xfence_unlock(ds);
}

static inline __device__ void load_int8_a_panel_ds(_global_ptr_ const int8_t* gm,
                                                   int mm, int packed_k, int a_ld,
                                                   v32i8* l1, int dmain,
                                                   int dst_pos, int ds) {
    dma_in_int8_dev(gm, mm, packed_k, a_ld, dmain, dst_pos);
    shuffle_int8_l2_to_l1d_ds(mm, packed_k, l1, ds);
}

static inline __device__ void load_int8_b_panel_ds(_global_ptr_ const int8_t* gm,
                                                   int packed_k, int nn, int b_ld,
                                                   v32i8* l1, int dmain,
                                                   int dst_pos, int ds) {
    dma_in_int8_dev(gm, packed_k, nn, b_ld, dmain, dst_pos);
    shuffle_int8_l2_to_l1w_ds(packed_k, nn, l1, ds);
}

static inline __device__ void store_l1e_fp32_ld(_global_ptr_ float* gm, int mm,
                                               int nn, int ld) {
    v16f32* l1e = (v16f32*)0;
    v16f32* l2e = (v16f32*)0;
    v16f32* l2r = (v16f32*)0;
    int nblocks = roundup_div16(nn);

    xfence_lock(EW);
    ew_cfg_activation_type(EW_NOACT);
    ew_cfg_coeff_scalar(1.0f, 0.0f, 0.0f);
    ew_cfg_findmax(0, 0.0f);
    ew_cfg_stream_size(mm * nblocks);
    ew_cfg_col_size(NBANKS);
    ew_dsmadd(l2e, l1e, l1e, EW_L2E);
    xfence_unlock(EW);

    xfence_lock(RS);
    rs_row_cfg_stride(1, nblocks);
    rs_row_cfg_loop(mm);
    for (int i = 0; i < nblocks; i++) {
        rs_row_batch(&l2r[i][0], &l2e[i * mm][0], NBANKS);
    }
    xfence_unlock(RS);

    dma_out_l2r_2d(gm, l2r, mm, nn, ld, roundup16(nn), (int)sizeof(float));
}

static inline __device__ void load_fp32_to_l1e_dst(_global_ptr_ const float* gm,
                                                   int mm, int nn, v16f32* l1e) {
    float* l2 = (float*)0;
    dma_in_1d(l2, gm, mm * nn * (int)sizeof(float), DMA_L2_0, DMA_FP32);
    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_FP32);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    float* l2_ptr = l2;
    for (int i = 0; i < nn; i += NBANKS) {
        ds_shuffle_coa(mm, &l1e[l1_iter][0], l2_ptr, DS_L1E);
        l2_ptr += NBANKS;
        l1_iter += mm;
    }
    xfence_unlock(DS_0);
}

static inline __device__ void reshape_l2e_to_hbm(_global_ptr_ float* gm, int mm, int nn) {
    v16f32* l2e = (v16f32*)0;
    v16f32* l2r = (v16f32*)0;
    int nblocks = roundup_div16(nn);
    xfence_lock(RS);
    rs_row_cfg_stride(1, nblocks);
    rs_row_cfg_loop(mm);
    for (int i = 0; i < nblocks; i++) {
        rs_row_batch(&l2r[i][0], &l2e[i * mm][0], NBANKS);
    }
    xfence_unlock(RS);
    dma_out_l2r_2d(gm, l2r, mm, nn, nn, roundup16(nn), (int)sizeof(float));
}

static inline __device__ void store_l1e_fp32_col(_global_ptr_ float* gm, int mm, int nn) {
    v16f32* l1e = (v16f32*)0;
    v16f32* l2e = (v16f32*)0;
    v16f32* l2r = (v16f32*)0;
    int nblocks = roundup_div16(nn);

    xfence_lock(EW);
    ew_cfg_activation_type(EW_NOACT);
    ew_cfg_coeff_scalar(1.0f, 0.0f, 0.0f);
    ew_cfg_findmax(0, 0.0f);
    ew_cfg_stream_size(mm * nblocks);
    ew_cfg_col_size(NBANKS);
    ew_dsmadd(l2e, l1e, l1e, EW_L2E);
    xfence_unlock(EW);

    int dst_row_stride = roundup_div16(mm);
    xfence_lock(RS);
    rs_col_cfg_loop(NBANKS);
    rs_col_cfg_stride(1, dst_row_stride, 1);
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < dst_row_stride; j++) {
            rs_col_batch(&l2r[j], &l2e[j * NBANKS], NBANKS);
        }
        l2r = &l2r[NBANKS * dst_row_stride];
        l2e = &l2e[mm];
    }
    xfence_unlock(RS);
    dma_out_l2r_2d(gm, (v16f32*)0, nn, mm, mm, roundup16(mm), (int)sizeof(float));
}

static inline __device__ void mac_int8_acc_twice(int k, int mm, int nn) {
    v32i8* l1d = (v32i8*)0;
    v32i8* l1w = (v32i8*)0;
    v16f32* l1e = (v16f32*)0;
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.0f);
    mm_cfg_basic_int8(k, mm, mm);
    mm_int8(l1e, l1d, l1w);
    mm_acc_int8(l1e, l1d, l1w);
    xfence_unlock(MAC);
    (void)nn;
}

static inline __device__ void mac_int16_acc_twice(int k, int mm, int nn) {
    v16i16* l1d = (v16i16*)0;
    v16i16* l1w = (v16i16*)0;
    v16f32* l1e = (v16f32*)0;
    int roundup_k = roundup16(k);
    int ntiles = roundup_div16(nn);
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.0f);
    mm_cfg_basic_int16(k, mm);
    for (int n_iter = 0; n_iter < ntiles; n_iter++) {
        mm_int16(&l1e[n_iter * mm], l1d, &l1w[n_iter * roundup_k]);
        mm_acc_int16(&l1e[n_iter * mm], l1d, &l1w[n_iter * roundup_k]);
    }
    xfence_unlock(MAC);
}

static inline __device__ float int31_scale(int bits) {
    float s = 0.0f;
    *(int*)(&s) = bits;
    return s;
}

static inline __device__ void load_int31_to_l1(_global_ptr_ const float* gm,
                                               int mm, int nn, v16i16* l1,
                                               int dst_sram, float max_val) {
    int16_t* l2_low = (int16_t*)0;
    int16_t* l2_high = (int16_t*)(L2DW_SIZE_PER_CORE / 2);
    int dest_off = (L2DW_SIZE_PER_CORE / 2) << 12;
    xfence_lock(DMAIN_0);
    dma_cfg((DMA_L2_0 | DMA_INT31 | dest_off),
            (DMA_GM | DMA_FP32 | highptr(gm)), max_val);
    dma_i(l2_low, lowptr(gm), mm * nn * (int)sizeof(float));
    xfence_unlock(DMAIN_0);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT16);
    ds_shuffle_cfg_blkx(nn);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int16_t* low_ptr = l2_low;
    int16_t* high_ptr = l2_high;
    int row_stride = roundup16(nn);
    for (int i = 0; i < mm; i += NBANKS) {
        ds_shuffle_batch(&l1[l1_iter][0], low_ptr, nn, dst_sram);
        low_ptr += NBANKS * nn;
        l1_iter += row_stride;
        ds_shuffle_batch(&l1[l1_iter][0], high_ptr, nn, dst_sram);
        high_ptr += NBANKS * nn;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

static inline __device__ void load_int31_to_l1_coa(_global_ptr_ const float* gm,
                                                   int nrows, int ncols, v16i16* l1,
                                                   int dst_sram, float max_val) {
    int16_t* l2_low = (int16_t*)0;
    int16_t* l2_high = (int16_t*)(L2DW_SIZE_PER_CORE / 2);
    int dest_off = (L2DW_SIZE_PER_CORE / 2) << 12;
    xfence_lock(DMAIN_0);
    dma_cfg((DMA_L2_0 | DMA_INT31 | dest_off),
            (DMA_GM | DMA_FP32 | highptr(gm)), max_val);
    dma_i(l2_low, lowptr(gm), nrows * ncols * (int)sizeof(float));
    xfence_unlock(DMAIN_0);

    xfence_lock(DS_0);
    ds_shuffle_cfg_datatype(DS_INT16);
    ds_shuffle_cfg_blkx(ncols);
    ds_cfg_output_bank(NBANKS);
    int l1_iter = 0;
    int16_t* low_ptr = l2_low;
    int16_t* high_ptr = l2_high;
    int row_stride = roundup16(nrows);
    for (int i = 0; i < ncols; i += NBANKS) {
        ds_shuffle_coa(nrows, &l1[l1_iter][0], low_ptr, dst_sram);
        low_ptr += NBANKS;
        l1_iter += row_stride;
        ds_shuffle_coa(nrows, &l1[l1_iter][0], high_ptr, dst_sram);
        high_ptr += NBANKS;
        l1_iter += row_stride;
    }
    xfence_unlock(DS_0);
}

static inline __device__ void mac_int31_tile(int k, int mm, int nn, float max_a, float max_b) {
    v16i16* l1d = (v16i16*)0;
    v16i16* l1w = (v16i16*)0;
    v16f32* l1e = (v16f32*)0;
    int rk = roundup16(k);
    float ll = max_a * max_b * int31_scale(0x21800000);
    float hl = max_a * max_b * int31_scale(0x29000000);
    float hh = max_a * max_b * int31_scale(0x30800000);
    xfence_lock(MAC);
    mm_cfg_stride(1);
    mm_cfg_basic_int16(k, mm);
    mm_cfg_dequant_scale(ll);
    mm_int16(l1e, l1d, l1w);
    mm_cfg_dequant_scale(hl);
    mm_acc_int16(l1e, &l1d[rk], l1w);
    mm_acc_int16(l1e, l1d, &l1w[rk]);
    mm_cfg_dequant_scale(hh);
    mm_acc_int16(l1e, &l1d[rk], &l1w[rk]);
    xfence_unlock(MAC);
    (void)nn;
}
