/*
 * csi_uart_out.c - control messages on the UART + NDJSON diagnostics (see csi_uart_out.h).
 *
 * Control message (UART pin; csi_control.c formats it):
 *   {"version":1,"event":"control","state":"on","reason":"prediction",
 *    "class":"approaching","confidence":0.87,"seq":42,"up_ms":123456,"crc":51234}
 *
 * Diagnostic frames (debug console; one per line; the crc field is optional for the host):
 *   {"version":1,"event":"presence","detected":true,"confidence":0.67,
 *    "model_latency_ms":0,"model_latency_us":180,"rssi":-58,"seq":1042,
 *    "up_ms":123456,"class":"walking","crc":41234}
 *   {"version":1,"event":"heartbeat","seq":1043,"up_ms":124456,"sent":900,
 *    "drops":0,"txerr":0,"fmt":0,"rej":0,"qmax":1,"crc":9911}
 *   {"version":1,"event":"error","seq":1044,"up_ms":130000,"code":"no_csi",
 *    "message":"no valid CSI records received","crc":777}
 *
 * The confidence is printed from integer percent, so no float printf support
 * is needed. "model_latency_ms" is kept for the v1 schema (rounded from us).
 *
 * The task waits on its queue only until the next control/heartbeat deadline,
 * so stale-prediction checks and the periodic control resend happen on time
 * even while results keep arriving. A blocking UART write of ~150 bytes
 * (~13 ms at 115200 baud) per second is harmless because it runs in this task,
 * not in the CSI path. Replace uart_tx() with a DMA transfer if the rate grows.
 */
#include "csi_uart_out.h"

#include <stdio.h>
#include <string.h>

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include "fsl_debug_console.h"

#include "csi_control.h"
#include "csi_presence.h"
#ifdef CSI_PRES_UART_BASE
#include "fsl_usart.h"
#ifndef CSI_PRES_UART_CLK_HZ
#ifdef CSI_PRES_UART_FLEXCOMM
#define CSI_PRES_UART_CLK_HZ CLOCK_GetFlexCommClkFreq(CSI_PRES_UART_FLEXCOMM)
#else
#error "CSI_PRES_UART_BASE needs CSI_PRES_UART_FLEXCOMM or CSI_PRES_UART_CLK_HZ"
#endif
#endif
#endif

typedef enum
{
    MSG_PRESENCE = 0,
    MSG_ERROR    = 1,
    MSG_FAULT    = 2
} msg_type_t;

typedef struct
{
    uint8_t type;
    union
    {
        csi_result_t res;
        uint8_t err;
    } u;
} csi_msg_t;

static const char *const s_err_code[CSI_ERR_COUNT] = {"no_csi", "model", "selftest"};
static const char *const s_err_msg[CSI_ERR_COUNT]  = {"no valid CSI records received",
                                                       "model returned an invalid class",
                                                       "self-test or init failed; control latched OFF"};

static QueueHandle_t s_q;
static char s_frame[CSI_OUT_MAX_FRAME];
static char s_ctl_frame[CSI_OUT_MAX_FRAME];
static uint32_t s_seq, s_ctl_seq;
static csi_ctrl_t s_ctl;
static volatile uint8_t s_fault_latch;
static volatile uint32_t s_sent, s_dropped, s_fmt_err, s_tx_err, s_max_depth, s_ctl_sent, s_ctl_stale_res;

static uint32_t up_ms(void)
{
    return (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
}

static void uart_tx(const char *p, size_t n)
{
#ifdef CSI_PRES_UART_BASE
    if (USART_WriteBlocking(CSI_PRES_UART_BASE, (const uint8_t *)p, n) != kStatus_Success)
    {
        s_tx_err++;
    }
#else
    (void)p;
    (void)n;
#endif
}

/* ---- control stream ------------------------------------------------------- */

static void send_control(void)
{
    int n = csi_ctrl_format(&s_ctl, s_ctl_seq, up_ms(), CSI_CTRL_CRC != 0, s_ctl_frame, sizeof(s_ctl_frame));
    if (n < 0)
    {
        /* never send a truncated line: fall back to OFF/fault, which always fits */
        s_fmt_err++;
        (void)csi_ctrl_on_fault(&s_ctl, false);
        n = csi_ctrl_format(&s_ctl, s_ctl_seq, up_ms(), CSI_CTRL_CRC != 0, s_ctl_frame, sizeof(s_ctl_frame));
        if (n < 0)
        {
            return;
        }
    }
#if CSI_OUT_UART_PROTOCOL == CSI_OUT_PROTO_CONTROL
    uart_tx(s_ctl_frame, (size_t)n);
#endif
#if CSI_CTRL_MIRROR_CONSOLE
    PRINTF("%s", s_ctl_frame);
#endif
    s_ctl_seq++;
    s_ctl_sent++;
}

/* ---- diagnostic stream ------------------------------------------------------ */

static void diag_tx(const char *p, size_t n)
{
#if CSI_OUT_UART_PROTOCOL == CSI_OUT_PROTO_DIAG
    uart_tx(p, n);
#else
    (void)n;
#endif
#if CSI_OUT_MIRROR_CONSOLE
    PRINTF("%s", p); /* p is NUL terminated by snprintf */
#else
    (void)p;
#endif
}

/* n = length of the frame body written so far (no closing brace yet). */
static void finish_and_send(int n)
{
    if ((n <= 0) || (n >= ((int)sizeof(s_frame) - 24)))
    {
        s_fmt_err++;
        return;
    }
#if CSI_OUT_CRC
    uint16_t crc = csi_crc16_ccitt((const uint8_t *)s_frame, (size_t)n);
    int m        = snprintf(&s_frame[n], sizeof(s_frame) - (size_t)n, ",\"crc\":%u}\n", (unsigned)crc);
#else
    int m = snprintf(&s_frame[n], sizeof(s_frame) - (size_t)n, "}\n");
#endif
    if ((m <= 0) || ((size_t)(n + m) >= sizeof(s_frame)))
    {
        s_fmt_err++;
        return;
    }
    diag_tx(s_frame, (size_t)(n + m));
    s_seq++;
    s_sent++;
}

static void send_presence(const csi_result_t *r)
{
    int n = snprintf(s_frame, sizeof(s_frame),
                     "{\"version\":1,\"event\":\"presence\",\"detected\":%s,\"confidence\":%u.%02u,"
                     "\"model_latency_ms\":%lu,\"model_latency_us\":%lu,\"rssi\":%d,\"seq\":%lu,"
                     "\"up_ms\":%lu,\"class\":\"%s\"",
                     (r->detected != 0U) ? "true" : "false", (unsigned)(r->conf_pct / 100U),
                     (unsigned)(r->conf_pct % 100U), (unsigned long)((r->latency_us + 500U) / 1000U),
                     (unsigned long)r->latency_us, (int)r->rssi, (unsigned long)s_seq, (unsigned long)up_ms(),
                     r->name);
    finish_and_send(n);
}

static void send_heartbeat(void)
{
    int n = snprintf(s_frame, sizeof(s_frame),
                     "{\"version\":1,\"event\":\"heartbeat\",\"seq\":%lu,\"up_ms\":%lu,\"sent\":%lu,"
                     "\"drops\":%lu,\"txerr\":%lu,\"fmt\":%lu,\"rej\":%lu,\"qmax\":%lu",
                     (unsigned long)s_seq, (unsigned long)up_ms(), (unsigned long)s_sent,
                     (unsigned long)s_dropped, (unsigned long)s_tx_err, (unsigned long)s_fmt_err,
                     (unsigned long)csi_presence_rejected(), (unsigned long)s_max_depth);
    finish_and_send(n);
}

static void send_error(uint8_t code)
{
    if (code >= (uint8_t)CSI_ERR_COUNT)
    {
        return;
    }
    int n = snprintf(s_frame, sizeof(s_frame),
                     "{\"version\":1,\"event\":\"error\",\"seq\":%lu,\"up_ms\":%lu,\"code\":\"%s\","
                     "\"message\":\"%s\"",
                     (unsigned long)s_seq, (unsigned long)up_ms(), s_err_code[code], s_err_msg[code]);
    finish_and_send(n);
}

/* ---- task --------------------------------------------------------------------- */

static void handle(const csi_msg_t *m, uint32_t now)
{
    bool changed = false;

    if (m->type == (uint8_t)MSG_PRESENCE)
    {
        send_presence(&m->u.res);
        if ((uint32_t)(now - m->u.res.t_ms) > CSI_CTRL_STALE_MS)
        {
            s_ctl_stale_res++; /* too old to act on; the stale check has already turned it OFF */
        }
        else
        {
            changed = csi_ctrl_on_prediction(&s_ctl, m->u.res.action, m->u.res.name, m->u.res.conf_pct,
                                             m->u.res.t_ms);
        }
    }
    else
    {
        send_error(m->u.err);
        if (m->type == (uint8_t)MSG_FAULT)
        {
            s_fault_latch = 1U;
        }
        if ((m->type == (uint8_t)MSG_FAULT) || (m->u.err == (uint8_t)CSI_ERR_MODEL))
        {
            (void)csi_ctrl_on_fault(&s_ctl, m->type == (uint8_t)MSG_FAULT);
            changed = true; /* report the fault right away even if already OFF */
        }
    }
    if (changed)
    {
        send_control();
    }
}

static uint32_t ms_until(uint32_t last, uint32_t period, uint32_t now)
{
    uint32_t elapsed = now - last;
    return (elapsed >= period) ? 0U : (period - elapsed);
}

static void out_task(void *arg)
{
    csi_msg_t m;
    uint8_t no_csi_reported = 0U;
    uint32_t now            = up_ms();
    uint32_t last_ctl = now, last_hb = now;

    (void)arg;
    csi_ctrl_init(&s_ctl, now);
    send_control(); /* OFF/startup as soon as the UART is up */

    for (;;)
    {
        now           = up_ms();
        uint32_t wait = ms_until(last_ctl, CSI_CTRL_PERIOD_MS, now);
        uint32_t hb   = ms_until(last_hb, CSI_OUT_HEARTBEAT_MS, now);
        if (hb < wait)
        {
            wait = hb;
        }
        if (s_ctl.reason == (uint8_t)CSI_CTRL_REASON_PREDICTION)
        {
            uint32_t st = ms_until(s_ctl.last_pred_ms, CSI_CTRL_STALE_MS + 1U, now); /* wake exactly when stale */
            if (st < wait)
            {
                wait = st;
            }
        }
        if (xQueueReceive(s_q, &m, pdMS_TO_TICKS(wait)) == pdTRUE)
        {
            uint32_t depth = (uint32_t)uxQueueMessagesWaiting(s_q) + 1U;
            if (depth > s_max_depth)
            {
                s_max_depth = depth;
            }
            handle(&m, up_ms());
        }

        now = up_ms();
        if ((s_fault_latch != 0U) && (s_ctl.fault_latched == 0U))
        {
            (void)csi_ctrl_on_fault(&s_ctl, true); /* posted while the queue was full */
            send_control();
            last_ctl = now;
        }
        if (csi_ctrl_poll(&s_ctl, now, CSI_CTRL_STALE_MS, CSI_CTRL_STARTUP_GRACE_MS))
        {
            send_control();
            last_ctl = now;
        }
        if ((uint32_t)(now - last_ctl) >= CSI_CTRL_PERIOD_MS)
        {
            send_control();
            last_ctl = now;
        }
        if ((uint32_t)(now - last_hb) >= CSI_OUT_HEARTBEAT_MS)
        {
            send_heartbeat();
            last_hb = now;
        }
        if (csi_presence_ms_since_feed() > CSI_OUT_NO_DATA_MS)
        {
            if (no_csi_reported == 0U)
            {
                no_csi_reported = 1U;
                send_error((uint8_t)CSI_ERR_NO_CSI);
            }
        }
        else
        {
            no_csi_reported = 0U;
        }
    }
}

bool csi_out_post_result(const csi_result_t *r)
{
    csi_msg_t m;
    m.type  = (uint8_t)MSG_PRESENCE;
    m.u.res = *r;
    if ((s_q == NULL) || (xQueueSend(s_q, &m, 0) != pdTRUE))
    {
        s_dropped++;
        return false;
    }
    return true;
}

static void post_err(uint8_t type, csi_err_t code)
{
    csi_msg_t m;
    m.type  = type;
    m.u.err = (uint8_t)code;
    if ((s_q == NULL) || (xQueueSend(s_q, &m, 0) != pdTRUE))
    {
        s_dropped++;
    }
}

void csi_out_post_error(csi_err_t code)
{
    post_err((uint8_t)MSG_ERROR, code);
}

void csi_out_post_fault(csi_err_t code)
{
    s_fault_latch = 1U;
    post_err((uint8_t)MSG_FAULT, code);
}

int csi_out_uart_reinit(void)
{
#ifdef CSI_PRES_UART_BASE
    usart_config_t cfg;
    USART_GetDefaultConfig(&cfg);
    cfg.baudRate_Bps = CSI_PRES_UART_BAUD;
    cfg.enableTx     = true;
    cfg.enableRx     = false;
    if (USART_Init(CSI_PRES_UART_BASE, &cfg, CSI_PRES_UART_CLK_HZ) != kStatus_Success)
    {
        return -1;
    }
#endif
    return 0;
}

int csi_out_init(void)
{
    if (s_q != NULL)
    {
        return 0; /* already initialised */
    }
    if (csi_out_uart_reinit() != 0)
    {
        return -1;
    }
    s_q = xQueueCreate(CSI_OUT_QUEUE_DEPTH, sizeof(csi_msg_t));
    if (s_q == NULL)
    {
        return -1;
    }
    if (xTaskCreate(out_task, "csi_out", 1024U, NULL, tskIDLE_PRIORITY + 1U, NULL) != pdPASS)
    {
        vQueueDelete(s_q);
        s_q = NULL;
        return -1;
    }
    return 0;
}
