/*
 * csi_presence.c - preprocessing + featurize + exported forest model.
 *
 * Mirrors csi_clean.py --window-mode records (record_fields, tsf_resets,
 * stream_window_starts, decode_amp_f32, normalize_rows_c) and
 * csi_train_eval.featurize(); change both sides together.
 *   1. validate record (structure, payload size), keep only the training AP
 *      (CSI_AP_MAC) on the training channel (CSI_CHANNEL)
 *   2. a TSF jump > CSI_STALL_US or a TSF going backwards empties the window
 *   3. decode I/Q -> amplitude, keep CSI_KEPT_SC, divide by the row mean
 *   [Hampel is NOT ported: training uses --hampel-window 1.]
 *   4. ring of CSI_WIN_LEN records; once full, classify every CSI_HOP records
 *   5. features in chronological order (CSI_FEATURE_SET), csi_infer()
 *   6. optional vote, map class -> ON/OFF/HOLD action, csi_out_post_result()
 */
#include "csi_presence.h"

#include <math.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"
#include "fsl_debug_console.h"
#include "fsl_device_registers.h"

#include "csi_model_config.h"
#include "csi_model.h" /* forest tables + csi_infer(); include from this file only */
#include "csi_uart_out.h"

#define HDR_BYTES  48U
#define TAIL_BYTES 4U

/* Feature block positions per CSI_FEATURE_SET (-1 = not present). */
#if CSI_FEATURE_SET == 0
#define FB_MEAN 0
#define FB_VAR 1
#define FB_SLOPE (-1)
#define FB_ABSDIFF (-1)
#define FB_COUNT 2U
#elif CSI_FEATURE_SET == 1
#define FB_MEAN (-1)
#define FB_VAR 0
#define FB_SLOPE 1
#define FB_ABSDIFF 2
#define FB_COUNT 3U
#elif CSI_FEATURE_SET == 2
#define FB_MEAN 0
#define FB_VAR 1
#define FB_SLOPE 2
#define FB_ABSDIFF 3
#define FB_COUNT 4U
#else
#error "unknown CSI_FEATURE_SET in csi_model_config.h"
#endif

#if (FB_COUNT * CSI_N_SC) != CSI_N_FEATURES
#error "CSI_N_FEATURES does not match CSI_FEATURE_SET and CSI_N_SC"
#endif
#if CSI_N_CLASSES > 255U
#error "too many classes"
#endif

static float s_win[CSI_WIN_LEN][CSI_N_SC];
static float s_feat[CSI_N_FEATURES];
static uint16_t s_head, s_fill, s_since_hop;
static uint64_t s_last_tsf;
static uint8_t s_have_tsf;
static uint8_t s_enabled;
#if CSI_VOTE_N > 1U
static uint8_t s_hist[CSI_VOTE_N];
static uint8_t s_hist_n, s_hist_pos;
#endif
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

static uint32_t now_ms(void)
{
    return (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
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
#if CSI_MEASURE_LATENCY
    uint32_t mhz = SystemCoreClock / 1000000U;
    return (mhz != 0U) ? (c / mhz) : 0U;
#else
    (void)c;
    return 0U;
#endif
}

/* Same as csi_train_eval.featurize(): per subcarrier over the window, rows read
 * oldest first starting at ring row 'start':
 *   mean, population variance, slope = sum((x-mean)*tc)/sum(tc^2) with
 *   tc = t-(W-1)/2, absdiff = mean |x[t]-x[t-1]|. */
static void featurize(const float *win, uint32_t start, float *feat)
{
#if (FB_SLOPE >= 0)
    float tss = 0.0f;
    for (uint32_t t = 0U; t < CSI_WIN_LEN; t++)
    {
        float tc = (float)t - ((float)(CSI_WIN_LEN - 1U) * 0.5f);
        tss += tc * tc;
    }
#endif
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        float m = 0.0f;
        for (uint32_t t = 0U; t < CSI_WIN_LEN; t++)
        {
            m += win[(((start + t) % CSI_WIN_LEN) * CSI_N_SC) + k];
        }
        m /= (float)CSI_WIN_LEN;

        float v = 0.0f, s = 0.0f, ad = 0.0f, prev = 0.0f;
        for (uint32_t t = 0U; t < CSI_WIN_LEN; t++)
        {
            float x = win[(((start + t) % CSI_WIN_LEN) * CSI_N_SC) + k];
            float d = x - m;
            v += d * d;
            s += d * ((float)t - ((float)(CSI_WIN_LEN - 1U) * 0.5f));
            if (t > 0U)
            {
                ad += fabsf(x - prev);
            }
            prev = x;
        }
#if (FB_MEAN >= 0)
        feat[(FB_MEAN * CSI_N_SC) + k] = m;
#endif
        feat[(FB_VAR * CSI_N_SC) + k] = v / (float)CSI_WIN_LEN;
#if (FB_SLOPE >= 0)
        feat[(FB_SLOPE * CSI_N_SC) + k] = s / tss;
#else
        (void)s;
#endif
#if (FB_ABSDIFF >= 0)
        feat[(FB_ABSDIFF * CSI_N_SC) + k] = ad / (float)(CSI_WIN_LEN - 1U);
#else
        (void)ad;
#endif
    }
}

#if CSI_VOTE_N > 1U
/* Majority vote over the last CSI_VOTE_N results; ties go to the newest result.
 * *share_pct = percentage of those results agreeing with the winner. */
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

    int best = cls;
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
#endif

static void reset_stream(void)
{
    s_head = s_fill = s_since_hop = 0U;
#if CSI_VOTE_N > 1U
    s_hist_n = s_hist_pos = 0U;
#endif
}

static void classify_and_report(void)
{
    uint32_t t0 = cycles();
    /* when full, s_head is the oldest row */
    featurize(&s_win[0][0], (uint32_t)s_head, s_feat);
#ifdef CSI_PRES_FEATURE_HOOK
    CSI_PRES_FEATURE_HOOK(s_feat, CSI_N_FEATURES);
#endif
    uint8_t conf = 0U;
    int32_t cls  = csi_infer(s_feat, (int32_t)CSI_N_FEATURES, &conf);
    uint32_t us  = cycles_to_us(cycles() - t0);

    if ((cls < 0) || (cls >= (int32_t)CSI_N_CLASSES) || (conf > 100U))
    {
        csi_out_post_error(CSI_ERR_MODEL);
        return;
    }

#if CSI_VOTE_N > 1U
    int smooth = vote((int)cls, &conf);
#else
    int smooth = (int)cls;
#endif

    csi_result_t res;
    res.name       = CSI_CLASS_NAME[smooth];
    res.cls        = (uint8_t)smooth;
    res.action     = CSI_CLASS_ACTION[smooth];
    res.detected   = (smooth != (int)CSI_EMPTY_CLASS) ? 1U : 0U; /* no "empty" class -> always 1 */
    res.conf_pct   = conf;
    res.rssi       = (s_rssi_n > 0U) ? (int8_t)(s_rssi_sum / (int32_t)s_rssi_n) : (int8_t)0;
    res.latency_us = us;
    res.t_ms       = now_ms();
    s_rssi_sum     = 0;
    s_rssi_n       = 0U;

#ifdef CSI_PRES_OUT_HOOK
    CSI_PRES_OUT_HOOK(smooth);
#endif
    (void)csi_out_post_result(&res);
}

void csi_presence_feed(const uint8_t *r, size_t len)
{
    if ((r == NULL) || (s_enabled == 0U))
    {
        return;
    }

    /* same structural checks as csi_clean.validate_row() */
    if ((len < (HDR_BYTES + TAIL_BYTES)) || (rd16(r + 2) != 0xABCDU) || (((size_t)rd16(r) * 4U) != len))
    {
        s_rejected++;
        return;
    }
    size_t dwords = (size_t)rd16(r + 44);
    if (dwords == 0U)
    {
        s_rejected++;
        return;
    }
    size_t data_bytes = (dwords - 1U) * 4U;
    if ((HDR_BYTES + data_bytes + TAIL_BYTES) != len)
    {
        s_rejected++;
        return;
    }
    /* radio filter, same order as csi_clean.apply_radio_filter(); not counted as rejected */
#if CSI_HAVE_AP_MAC
    if (memcmp(r + 26, CSI_AP_MAC, 6U) != 0)
    {
        return;
    }
#endif
    if (r[37] != (uint8_t)CSI_CHANNEL)
    {
        return;
    }
    if (data_bytes != (CSI_RAW_N_SC * 2U))
    {
        s_rejected++; /* different bandwidth/packet type than the training data */
        return;
    }

    s_last_feed_tick = xTaskGetTickCount();
    s_rssi_sum += (int32_t)(int8_t)r[32]; /* RSSI chain A */
    s_rssi_n++;

    /* a TSF gap or reset discards the partially filled window (tsf_resets()) */
    uint64_t tsf = (uint64_t)rd32(r + 12) | ((uint64_t)rd32(r + 16) << 32);
    if ((s_have_tsf != 0U) && ((tsf < s_last_tsf) || ((tsf - s_last_tsf) > CSI_STALL_US)))
    {
        reset_stream();
    }
    s_last_tsf = tsf;
    s_have_tsf = 1U;

    /* amplitude of kept subcarriers, normalized by their mean (normalize_rows_c()) */
    const uint8_t *iq = r + HDR_BYTES;
    float *row        = s_win[s_head];
    float sum         = 0.0f;
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        uint32_t i = CSI_KEPT_SC[k];
        float re   = (float)(int8_t)iq[2U * i] * CSI_IQ_SCALE;
        float im   = (float)(int8_t)iq[(2U * i) + 1U] * CSI_IQ_SCALE;
        float a    = sqrtf((re * re) + (im * im));
        row[k]     = a;
        sum += a;
    }
    float inv = 1.0f / ((sum / (float)CSI_N_SC) + 1e-9f);
    for (uint32_t k = 0U; k < CSI_N_SC; k++)
    {
        row[k] *= inv;
    }

    /* ring buffer, classify every CSI_HOP records once full (stream_window_starts()) */
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

/* Feeds cleaned (chronological) windows exported from Python through featurize
 * + model. Verifies the C feature code and the model, not I/Q decoding or the
 * streaming logic (csi_replay_parity.py checks those on the host). */
static int selftest(void)
{
    int fails = 0;
    for (int i = 0; i < CSI_SELFTEST_N; i++)
    {
        featurize(&CSI_SELFTEST_WIN[(uint32_t)i * CSI_WIN_LEN * CSI_N_SC], 0U, s_feat);

        float maxerr = 0.0f;
        for (uint32_t j = 0U; j < CSI_N_FEATURES; j++)
        {
            float ref = CSI_SELFTEST_FEAT[((uint32_t)i * CSI_N_FEATURES) + j];
            float e   = fabsf(s_feat[j] - ref) / (1.0f + fabsf(ref));
            if (e > maxerr)
            {
                maxerr = e;
            }
        }
        uint8_t conf = 0U;
        /* classify the reference features so a borderline float difference
         * cannot flip a tree split; the feature error is checked separately */
        int32_t cls = csi_infer(&CSI_SELFTEST_FEAT[(uint32_t)i * CSI_N_FEATURES], (int32_t)CSI_N_FEATURES, &conf);
        int ok      = (cls == (int32_t)CSI_SELFTEST_CLASS[i]) && (maxerr < 1e-4f);
        fails += ok ? 0 : 1;
        PRINTF("PRES_SELFTEST,%d,%s,c_class=%d,py_class=%d,relerr_e6=%d\r\n", i, ok ? "OK" : "FAIL", (int)cls,
               (int)CSI_SELFTEST_CLASS[i], (int)(maxerr * 1e6f));
    }
    return fails;
}
#endif

int csi_presence_init(void)
{
    s_enabled = 0U;
    cycles_init();
    reset_stream();
    s_have_tsf       = 0U;
    s_last_feed_tick = xTaskGetTickCount();
    if (csi_out_init() != 0)
    {
        return -1; /* no output task: the caller must keep the receiver OFF */
    }
#if CSI_PRESENCE_SELFTEST
    if (selftest() != 0)
    {
        csi_out_post_fault(CSI_ERR_SELFTEST);
        return -1;
    }
#endif
    s_enabled = 1U;
    return 0;
}
