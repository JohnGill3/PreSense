/*
 * csi_raw_log.c - log complete raw CSI records over the debug console.
 *
 * The Wi-Fi driver callback only validates and copies each record into a
 * static pool. A low-priority task prints it, so the driver is never blocked
 * by UART output.
 *
 * Output, one line per record (parse this on the PC):
 *     CSI,<seq>,<tick_ms>,<record_bytes>,<hex of the whole record>
 * The record is exactly what AN14281 section 3 describes:
 *     48-byte header | CSI data | 4-byte tail ID
 *
 * Diagnostics (only printed when something goes wrong):
 *     CSI_DROP,<total>            pool full or record larger than the limit
 *     CSI_BAD,<total>,len=..,rec=..   bad signature/length in the buffer
 *
 * Header byte offsets (little-endian, from AN14281 Table 2 and its example):
 *     0  length in dwords (16 bit)      2  signature 0xABCD
 *     4  header ID                      8  PKT_info
 *    12  TSF (64 bit, low dword first)  20  dst MAC (6 bytes)   26  src MAC
 *    32  RSSI A   33 RSSI B   34 NF A   35 NF B      (int8)
 *    36  SINR (int8)  37 channel  38 AP type  39 chip ID
 *    40  FCF (16 bit) 42 total gain
 *    44  CSI data length in dwords + 1 (16 bit)
 *    48  CSI data (I,Q int8 pairs per subcarrier), then 4-byte tail ID
 */
#include "csi_raw_log.h"

#include <stdio.h>
#include <string.h>

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include "fsl_debug_console.h"

#define CSI_SIGNATURE   0xABCDU
#define CSI_HDR_BYTES   48U
#define CSI_TAIL_BYTES  4U
#define CSI_MIN_RECORD  (CSI_HDR_BYTES + CSI_TAIL_BYTES)

typedef struct
{
    uint16_t len; /* record length in bytes */
    uint8_t data[CSI_LOG_MAX_RECORD_BYTES];
} csi_item_t;

static csi_item_t s_pool[CSI_LOG_QUEUE_DEPTH];
static QueueHandle_t s_free; /* indices of unused pool entries */
static QueueHandle_t s_full; /* indices waiting to be printed  */
static char s_line[64U + 2U * CSI_LOG_MAX_RECORD_BYTES];

static volatile uint32_t s_dropped;
static volatile uint32_t s_bad;
static volatile uint32_t s_bad_len;
static volatile uint32_t s_bad_rec;

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

#if CSI_LOG_PRINT_SUMMARY
static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
#endif

/*
 * Runs in the Wi-Fi driver's context (assumed to be a task, not an ISR).
 * `len` is expected to be the size in bytes; the header length is used as the
 * authority, and a mismatch is reported through CSI_BAD.
 */
int csi_raw_log_cb(void *buffer, size_t len)
{
    const uint8_t *p = (const uint8_t *)buffer;
    size_t left      = len;

    if ((s_free == NULL) || (p == NULL))
    {
        return 0;
    }

    /* A buffer may hold more than one record; walk through all of them. */
    while (left >= CSI_MIN_RECORD)
    {
        size_t rec = (size_t)rd16(p) * 4U; /* length field is in dwords */
        uint8_t idx;

        if ((rd16(p + 2) != CSI_SIGNATURE) || (rec < CSI_MIN_RECORD) || (rec > left))
        {
            s_bad_len = (uint32_t)len;
            s_bad_rec = (uint32_t)rec;
            s_bad++;
            break;
        }

        if (rec > CSI_LOG_MAX_RECORD_BYTES)
        {
            s_dropped++;
        }
        else if (xQueueReceive(s_free, &idx, 0) == pdTRUE)
        {
            s_pool[idx].len = (uint16_t)rec;
            memcpy(s_pool[idx].data, p, rec);
            if (xQueueSend(s_full, &idx, 0) != pdTRUE)
            {
                s_dropped++;
                (void)xQueueSend(s_free, &idx, 0);
            }
        }
        else
        {
            s_dropped++; /* logger task can't keep up */
        }

        p += rec;
        left -= rec;
    }

    return 0;
}

#if CSI_LOG_PRINT_SUMMARY
static void print_summary(uint32_t seq, const csi_item_t *it)
{
    const uint8_t *r = it->data;

    PRINTF("CSIH,%lu,ch=%u,rssi=%d/%d,nf=%d/%d,sinr=%d,ap_type=%u,tsf=%08lx%08lx,"
           "src=%02x:%02x:%02x:%02x:%02x:%02x,data_dwords=%u\r\n",
           (unsigned long)seq, (unsigned)r[37], (int)(int8_t)r[32], (int)(int8_t)r[33], (int)(int8_t)r[34],
           (int)(int8_t)r[35], (int)(int8_t)r[36], (unsigned)r[38], (unsigned long)rd32(r + 16),
           (unsigned long)rd32(r + 12), (unsigned)r[26], (unsigned)r[27], (unsigned)r[28], (unsigned)r[29],
           (unsigned)r[30], (unsigned)r[31], (unsigned)(rd16(r + 44) - 1U));
}
#endif

static void print_record(uint32_t seq, const csi_item_t *it)
{
    static const char hexd[] = "0123456789abcdef";
    int n = snprintf(s_line, sizeof(s_line), "CSI,%lu,%lu,%u,", (unsigned long)seq,
                     (unsigned long)(xTaskGetTickCount() * portTICK_PERIOD_MS), (unsigned)it->len);
    char *o = s_line + n;

    for (uint16_t i = 0U; i < it->len; i++)
    {
        *o++ = hexd[it->data[i] >> 4];
        *o++ = hexd[it->data[i] & 0x0FU];
    }
    *o++ = '\r';
    *o++ = '\n';
    *o   = '\0';

    PRINTF("%s", s_line);
#if CSI_LOG_PRINT_SUMMARY
    print_summary(seq, it);
#endif
}

static void csi_log_task(void *arg)
{
    uint8_t idx;
    uint32_t seq = 0U, last_drop = 0U, last_bad = 0U;

    (void)arg;
    for (;;)
    {
        if (xQueueReceive(s_full, &idx, portMAX_DELAY) != pdTRUE)
        {
            continue;
        }
        print_record(seq++, &s_pool[idx]);
        (void)xQueueSend(s_free, &idx, 0);

        if (s_dropped != last_drop)
        {
            last_drop = s_dropped;
            PRINTF("CSI_DROP,%lu\r\n", (unsigned long)last_drop);
        }
        if (s_bad != last_bad)
        {
            last_bad = s_bad;
            PRINTF("CSI_BAD,%lu,len=%lu,rec=%lu\r\n", (unsigned long)last_bad, (unsigned long)s_bad_len,
                   (unsigned long)s_bad_rec);
        }
    }
}

int csi_raw_log_init(void)
{
    if (s_full != NULL)
    {
        return 0; /* already initialised */
    }

    s_free = xQueueCreate(CSI_LOG_QUEUE_DEPTH, sizeof(uint8_t));
    s_full = xQueueCreate(CSI_LOG_QUEUE_DEPTH, sizeof(uint8_t));
    if ((s_free == NULL) || (s_full == NULL))
    {
        return -1;
    }

    for (uint8_t i = 0U; i < CSI_LOG_QUEUE_DEPTH; i++)
    {
        (void)xQueueSend(s_free, &i, 0);
    }

    if (xTaskCreate(csi_log_task, "csi_log", 1024U, NULL, tskIDLE_PRIORITY + 1U, NULL) != pdPASS)
    {
        return -1;
    }
    return 0;
}
