/*
 * csi_presence.c - preprocessing + featurize + emlearn model.
 *
 * Mirrors the Python pipeline (csi_clean.py -> csi_train_export.py):
 *   1. validate record          (Phase 1)
 *   2. decode I/Q -> amplitude  (Phase 3), keep only non-null subcarriers (Phase 4)
 *   3. divide by record mean    (Phase 5)
 *   4. drop window on TSF gap   (Phase 6)
 *   [Hampel (Phase 7) is NOT ported: run csi_clean.py with --hampel-window 1
 *    so training data matches what the board computes.]
 *   5. window of CSI_WIN_LEN records, new result every CSI_HOP records (Phase 8)
 *   6. features = [mean per subcarrier | variance per subcarrier]
 *   7. csi_model_predict(), majority vote, then csi_out_post_result()
 *
 * Results leave through csi_uart_out.c (NDJSON on the UART pin and/or console).
 */
#include "csi_presence.h"

#include <math.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"
#include "fsl_debug_console.h"
#include "fsl_device_registers.h"

#include "csi_model_config.h"
#include "csi_model.h" /* emlearn: csi_model_predict() */
#include "csi_uart_out.h"

#define HDR_BYTES  48U
#define TAIL_BYTES 4U

static float s_win[CSI_WIN_LEN][CSI_N_SC];
static float s_feat[CSI_N_FEATURES];
static uint16_t s_head, s_fill, s_since_hop;
static uint64_t s_last_tsf;
static uint8_t s_have_tsf;
static uint8_t s_hist[CSI_VOTE_N];
static uint8_t s_hist_n, s_hist_pos;
static int32_t s_rssi_sum;
static uint32_t s_rssi_n;
static volatile TickType_t s_last_feed_tick;
static volatile uint32_t s_rejected;

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ---- latency measurement with the Cortex-M cycle counter -------------------
 * Set CSI_MEASURE_LATENCY 0 if the core has no DWT cycle counter; the latency
 * fields then read 0. The value is wall time, so it includes preemption. */
#ifndef CSI_MEASURE_LATENCY
#define CSI_MEASURE_LATENCY 1
#endif

static void cycles_init(void)
{
#if CSI_MEASURE_LATENCY
#if defined(DCB_DEMCR_TRCENA_Msk)
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
#else
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#endif
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
#endif
}

static uint32_t cycles(void)
{
#if CSI_MEASURE_LATENCY
    return DWT->CYCCNT;
#else
    return 0U;
#endif
}

static uint32_t cycles_to_us(uint32_t c)
{
    uint32_t mhz = SystemCoreClock / 1000000U;
    return (mhz != 0U) ? (c / mhz) : 0U;
}

/* Same as Python featurize(): mean over time, then population variance over time.
 * Feature order: [mean_0..mean_{N-1}, var_0..var_{N-1}]. */
static void featurize(const float (*win)[CSI_N_SC], float *feat)
{
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        float m = 0.0f;
        for (uint32_t t = 0U; t < CSI_WIN_LEN; t++)
        {
            m += win[t][k];
        }
        m /= (float)CSI_WIN_LEN;

        float v = 0.0f;
        for (uint32_t t = 0U; t < CSI_WIN_LEN; t++)
        {
            float d = win[t][k] - m;
            v += d * d;
        }
        feat[k]            = m;
        feat[CSI_N_SC + k] = v / (float)CSI_WIN_LEN;
    }
}

static int predict(const float *feat)
{
    /* Check the generated csi_model.h: if csi_model_predict() takes a different
     * argument type than const float*, convert the features here. */
    return (int)csi_model_predict(feat, CSI_N_FEATURES);
}

/* Majority vote over the last CSI_VOTE_N results. *share_pct is the percentage
 * of those results that agree with the winner (used as "confidence"). */
static int vote(int cls, uint8_t *share_pct)
{
    uint8_t counts[CSI_N_CLASSES];
    memset(counts, 0, sizeof(counts));

    s_hist[s_hist_pos] = (uint8_t)cls;
    s_hist_pos         = (uint8_t)((s_hist_pos + 1U) % CSI_VOTE_N);
    if (s_hist_n < CSI_VOTE_N)
    {
        s_hist_n++;
    }
    for (uint8_t i = 0U; i < s_hist_n; i++)
    {
        counts[s_hist[i]]++;
    }

    int best = cls; /* ties go to the newest result */
    for (int c = 0; c < (int)CSI_N_CLASSES; c++)
    {
        if (counts[c] > counts[best])
        {
            best = c;
        }
    }
    *share_pct = (uint8_t)(((uint32_t)counts[best] * 100U) / s_hist_n);
    return best;
}

static void classify_and_report(void)
{
    uint32_t t0 = cycles();
    /* mean and variance do not depend on row order, so the ring needs no unrolling */
    featurize((const float (*)[CSI_N_SC])s_win, s_feat);
    int cls     = predict(s_feat);
    uint32_t us = cycles_to_us(cycles() - t0);

    if ((cls < 0) || (cls >= (int)CSI_N_CLASSES))
    {
        csi_out_post_error(CSI_ERR_MODEL);
        return;
    }

    uint8_t share = 0U;
    int smooth    = vote(cls, &share);

    csi_result_t res;
    res.name       = CSI_CLASS_NAME[smooth];
    res.cls        = (uint8_t)smooth;
    res.detected   = (smooth != (int)CSI_EMPTY_CLASS) ? 1U : 0U; /* no "empty" class -> always 1 */
    res.conf_pct   = share;
    res.rssi       = (s_rssi_n > 0U) ? (int8_t)(s_rssi_sum / (int32_t)s_rssi_n) : (int8_t)0;
    res.latency_us = us;
    s_rssi_sum     = 0;
    s_rssi_n       = 0U;

    (void)csi_out_post_result(&res);
}

void csi_presence_feed(const uint8_t *r, size_t len)
{
    if (r == NULL)
    {
        return;
    }

    /* Phase 1: same checks as the Python validator */
    if ((len < (HDR_BYTES + TAIL_BYTES)) || (rd16(r + 2) != 0xABCDU) || (((size_t)rd16(r) * 4U) != len))
    {
        s_rejected++;
        return;
    }
    size_t data_bytes = ((size_t)rd16(r + 44) - 1U) * 4U;
    if ((data_bytes != (CSI_RAW_N_SC * 2U)) || ((HDR_BYTES + data_bytes + TAIL_BYTES) != len))
    {
        s_rejected++; /* different bandwidth/packet type than the training data */
        return;
    }
#if CSI_HAVE_AP_MAC
    if (memcmp(r + 26, CSI_AP_MAC, 6U) != 0)
    {
        return; /* other transmitters are normal, not counted as rejected */
    }
#endif

    s_last_feed_tick = xTaskGetTickCount();
    s_rssi_sum += (int32_t)(int8_t)r[32]; /* RSSI chain A */
    s_rssi_n++;

    /* Phase 6: a gap in the TSF discards the partially filled window */
    uint64_t tsf = (uint64_t)rd32(r + 12) | ((uint64_t)rd32(r + 16) << 32);
    if (s_have_tsf && ((tsf < s_last_tsf) || ((tsf - s_last_tsf) > CSI_STALL_US)))
    {
        s_head = s_fill = s_since_hop = 0U;
    }
    s_last_tsf = tsf;
    s_have_tsf = 1U;

    /* Phases 3-5: amplitude of kept subcarriers, normalized by their mean */
    const uint8_t *iq = r + HDR_BYTES;
    float *row        = s_win[s_head];
    float sum         = 0.0f;
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        uint32_t i = CSI_KEPT_SC[k];
        float re   = (float)(int8_t)iq[2U * i] * CSI_IQ_SCALE;
        float im   = (float)(int8_t)iq[2U * i + 1U] * CSI_IQ_SCALE;
        float a    = sqrtf((re * re) + (im * im));
        row[k]     = a;
        sum += a;
    }
    float inv = 1.0f / ((sum / (float)CSI_N_SC) + 1e-9f);
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        row[k] *= inv;
    }

    /* Phase 8: ring buffer, classify every CSI_HOP records once it is full */
    s_head = (uint16_t)((s_head + 1U) % CSI_WIN_LEN);
    if (s_fill < CSI_WIN_LEN)
    {
        s_fill++;
    }
    s_since_hop++;

    if ((s_fill == CSI_WIN_LEN) && (s_since_hop >= CSI_HOP))
    {
        s_since_hop = 0U;
        classify_and_report();
    }
}

uint32_t csi_presence_ms_since_feed(void)
{
    return (uint32_t)((xTaskGetTickCount() - s_last_feed_tick) * portTICK_PERIOD_MS);
}

uint32_t csi_presence_rejected(void)
{
    return s_rejected;
}

#if CSI_PRESENCE_SELFTEST
#include "csi_selftest.h"

/* Feeds cleaned windows exported from Python through featurize + model.
 * Verifies the C feature code and the emlearn model, not the I/Q decoding. */
static int selftest(void)
{
    int fails = 0;
    for (int i = 0; i < CSI_SELFTEST_N; i++)
    {
        featurize((const float (*)[CSI_N_SC]) & CSI_SELFTEST_WIN[i * CSI_WIN_LEN * CSI_N_SC], s_feat);

        float maxerr = 0.0f;
        for (uint32_t j = 0U; j < CSI_N_FEATURES; j++)
        {
            float e = fabsf(s_feat[j] - CSI_SELFTEST_FEAT[(uint32_t)i * CSI_N_FEATURES + j]);
            if (e > maxerr)
            {
                maxerr = e;
            }
        }
        int cls = predict(s_feat);
        int ok  = (cls == (int)CSI_SELFTEST_CLASS[i]) && (maxerr < 1e-4f);
        fails += ok ? 0 : 1;
        PRINTF("PRES_SELFTEST,%d,%s,c_class=%d,py_class=%d,maxerr_e6=%d\r\n", i, ok ? "OK" : "FAIL", cls,
               (int)CSI_SELFTEST_CLASS[i], (int)(maxerr * 1e6f));
    }
    return fails;
}
#endif

int csi_presence_init(void)
{
    cycles_init();
    s_last_feed_tick = xTaskGetTickCount();
    if (csi_out_init() != 0)
    {
        return -1;
    }
#if CSI_PRESENCE_SELFTEST
    return (selftest() == 0) ? 0 : -1;
#else
    return 0;
#endif
}
