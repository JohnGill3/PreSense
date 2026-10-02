/*
 * csi_uart_out.c - formats PreSense NDJSON frames and sends them on the UART.
 *
 * Example frames (one per line; the crc field is optional for the host):
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
 * At about one presence frame per hop (~1 s) a blocking UART write of ~170
 * bytes (~15 ms at 115200 baud) is harmless because it runs in this task, not
 * in the CSI path. Replace tx() with a DMA transfer only if the rate grows.
 */
#include "csi_uart_out.h"

#include <stdio.h>
#include <string.h>

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include "fsl_debug_console.h"

#include "csi_presence.h"
#ifdef CSI_PRES_UART_BASE
#include "fsl_usart.h"
#endif

typedef enum
{
    MSG_PRESENCE = 0,
    MSG_ERROR    = 1
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

static const char *const s_err_code[CSI_ERR_COUNT] = {"no_csi", "model"};
static const char *const s_err_msg[CSI_ERR_COUNT]  = {"no valid CSI records received",
                                                       "model returned an invalid class"};

static QueueHandle_t s_q;
static char s_frame[CSI_OUT_MAX_FRAME];
static uint32_t s_seq;
static volatile uint32_t s_sent, s_dropped, s_fmt_err, s_tx_err, s_max_depth;

static uint32_t up_ms(void)
{
    return (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
}

#if CSI_OUT_CRC
/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor. */
static uint16_t crc16_ccitt(const uint8_t *d, size_t n)
{
    uint16_t crc = 0xFFFFU;
    while (n-- > 0U)
    {
        crc ^= (uint16_t)((uint16_t)(*d++) << 8);
        for (int i = 0; i < 8; i++)
        {
            crc = ((crc & 0x8000U) != 0U) ? (uint16_t)((crc << 1) ^ 0x1021U) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}
#endif

static void tx(const char *p, size_t n)
{
#ifdef CSI_PRES_UART_BASE
    if (USART_WriteBlocking(CSI_PRES_UART_BASE, (const uint8_t *)p, n) != kStatus_Success)
    {
        s_tx_err++;
    }
#endif
#if CSI_OUT_MIRROR_CONSOLE
    PRINTF("%s", p); /* p is NUL terminated by snprintf */
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
    uint16_t crc = crc16_ccitt((const uint8_t *)s_frame, (size_t)n);
    int m        = snprintf(&s_frame[n], sizeof(s_frame) - (size_t)n, ",\"crc\":%u}\n", (unsigned)crc);
#else
    int m = snprintf(&s_frame[n], sizeof(s_frame) - (size_t)n, "}\n");
#endif
    if ((m <= 0) || ((size_t)(n + m) >= sizeof(s_frame)))
    {
        s_fmt_err++;
        return;
    }
    tx(s_frame, (size_t)(n + m));
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

static void out_task(void *arg)
{
    csi_msg_t m;
    uint8_t no_csi_reported = 0U;

    (void)arg;
    for (;;)
    {
        if (xQueueReceive(s_q, &m, pdMS_TO_TICKS(CSI_OUT_HEARTBEAT_MS)) == pdTRUE)
        {
            uint32_t depth = (uint32_t)uxQueueMessagesWaiting(s_q) + 1U;
            if (depth > s_max_depth)
            {
                s_max_depth = depth;
            }
            if (m.type == (uint8_t)MSG_PRESENCE)
            {
                send_presence(&m.u.res);
            }
            else
            {
                send_error(m.u.err);
            }
        }
        else
        {
            /* Nothing to report for a whole heartbeat period. */
            send_heartbeat();

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

void csi_out_post_error(csi_err_t code)
{
    csi_msg_t m;
    m.type  = (uint8_t)MSG_ERROR;
    m.u.err = (uint8_t)code;
    if ((s_q == NULL) || (xQueueSend(s_q, &m, 0) != pdTRUE))
    {
        s_dropped++;
    }
}

int csi_out_init(void)
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
    s_q = xQueueCreate(CSI_OUT_QUEUE_DEPTH, sizeof(csi_msg_t));
    if (s_q == NULL)
    {
        return -1;
    }
    if (xTaskCreate(out_task, "csi_out", 1024U, NULL, tskIDLE_PRIORITY + 1U, NULL) != pdPASS)
    {
        return -1;
    }
    return 0;
}
