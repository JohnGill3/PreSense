/*
 * csi_uart_out.h - NDJSON output task for the PreSense wire protocol (version 1).
 *
 * One JSON object per line, max 256 bytes incl. '\n'. Events: presence,
 * heartbeat, error. Every frame carries "seq" (one counter for ALL events) and
 * "up_ms" (board uptime, so the host can tell a reboot from packet loss).
 *
 * Producers (csi_presence.c) only post small structs into a queue; this task
 * formats and transmits, so UART time never delays CSI processing.
 */
#ifndef CSI_UART_OUT_H_
#define CSI_UART_OUT_H_

#include <stdbool.h>
#include <stdint.h>

#ifndef CSI_OUT_QUEUE_DEPTH
#define CSI_OUT_QUEUE_DEPTH 8U
#endif

/* Protocol limit, including the terminating newline. */
#ifndef CSI_OUT_MAX_FRAME
#define CSI_OUT_MAX_FRAME 256U
#endif

/* Idle time after which a heartbeat is sent. */
#ifndef CSI_OUT_HEARTBEAT_MS
#define CSI_OUT_HEARTBEAT_MS 1000U
#endif

/* No valid CSI record for this long -> one "no_csi" error event. */
#ifndef CSI_OUT_NO_DATA_MS
#define CSI_OUT_NO_DATA_MS 5000U
#endif

/* 1 = append ,"crc":<CRC-16/CCITT-FALSE of everything before the comma> */
#ifndef CSI_OUT_CRC
#define CSI_OUT_CRC 1
#endif

/* 1 = also print every frame on the debug console (MCU-Link virtual COM port).
 * Handy for bring-up; set 0 once the console carries other logs. */
#ifndef CSI_OUT_MIRROR_CONSOLE
#define CSI_OUT_MIRROR_CONSOLE 1
#endif

/* The UART itself is enabled by defining, in the project settings:
 *     CSI_PRES_UART_BASE    e.g. USART0  (a spare FLEXCOMM routed to header pins)
 *     CSI_PRES_UART_CLK_HZ  clock frequency of that FLEXCOMM
 *     CSI_PRES_UART_BAUD    optional, default 115200
 * To use DMA instead of the blocking write, replace tx() in csi_uart_out.c.
 */
#ifndef CSI_PRES_UART_BAUD
#define CSI_PRES_UART_BAUD 115200U
#endif

typedef struct
{
    const char *name;    /* class name, plain [A-Za-z0-9_-] (goes into JSON unescaped) */
    uint8_t cls;         /* class index */
    uint8_t detected;    /* 1 = occupied */
    uint8_t conf_pct;    /* 0..100: share of the last votes agreeing with cls */
    int8_t rssi;         /* dBm, mean over the records since the previous result */
    uint32_t latency_us; /* featurize + predict time */
} csi_result_t;

typedef enum
{
    CSI_ERR_NO_CSI = 0,
    CSI_ERR_MODEL  = 1,
    CSI_ERR_COUNT
} csi_err_t;

/* Sets up the UART (if enabled), the queue and the output task. 0 = success. */
int csi_out_init(void);

/* Non-blocking. Returns false (and counts a drop) if the queue is full. */
bool csi_out_post_result(const csi_result_t *r);
void csi_out_post_error(csi_err_t code);

#endif /* CSI_UART_OUT_H_ */
