/*
 * csi_uart_out.h - output task: control UART + NDJSON diagnostics.
 *
 * Two streams, both formatted and sent by one low-priority task:
 *  - control (csi_control.h, schemas/presense-control.v1.schema.json): one
 *    JSON "control" message on every ON/OFF change and every
 *    CSI_CTRL_PERIOD_MS, on the dedicated UART (CSI_PRES_UART_BASE).
 *  - diagnostics (wire protocol version 1): presence, heartbeat, error. One
 *    JSON object per line, max 256 bytes incl. '\n'. Every frame carries "seq"
 *    (one counter for these events) and "up_ms" (board uptime, so the host can
 *    tell a reboot from packet loss). Sent on the debug console; on the UART
 *    pin only with CSI_OUT_UART_PROTOCOL = CSI_OUT_PROTO_DIAG (old behaviour).
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

/* Diagnostic heartbeat period on the console. */
#ifndef CSI_OUT_HEARTBEAT_MS
#define CSI_OUT_HEARTBEAT_MS 1000U
#endif

/* No valid CSI record for this long -> one "no_csi" error event. */
#ifndef CSI_OUT_NO_DATA_MS
#define CSI_OUT_NO_DATA_MS 5000U
#endif

/* 1 = append ,"crc":<CRC-16/CCITT-FALSE of everything before the comma> to diagnostic frames */
#ifndef CSI_OUT_CRC
#define CSI_OUT_CRC 1
#endif

/* 1 = print diagnostic frames on the debug console (MCU-Link virtual COM port). */
#ifndef CSI_OUT_MIRROR_CONSOLE
#define CSI_OUT_MIRROR_CONSOLE 1
#endif

/* What the UART pin (CSI_PRES_UART_BASE) carries. */
#define CSI_OUT_PROTO_CONTROL 0 /* only "control" messages (default) */
#define CSI_OUT_PROTO_DIAG    1 /* only v1 presence/heartbeat/error, as before */
#ifndef CSI_OUT_UART_PROTOCOL
#define CSI_OUT_UART_PROTOCOL CSI_OUT_PROTO_CONTROL
#endif

/* ---- control stream ---- */
/* Resend the current state this often even if unchanged. */
#ifndef CSI_CTRL_PERIOD_MS
#define CSI_CTRL_PERIOD_MS 1000U
#endif
/* No successful classification for this long -> OFF ("stale"). */
#ifndef CSI_CTRL_STALE_MS
#define CSI_CTRL_STALE_MS 5000U
#endif
/* No classification at all since boot for this long -> reason "stale" instead
 * of "startup" (window fill time plus CSI_CTRL_STALE_MS). */
#ifndef CSI_CTRL_STARTUP_GRACE_MS
#define CSI_CTRL_STARTUP_GRACE_MS 15000U
#endif
/* 1 = append "crc" to control messages (receivers should require it). */
#ifndef CSI_CTRL_CRC
#define CSI_CTRL_CRC 1
#endif
/* 1 = also print control messages on the debug console. */
#ifndef CSI_CTRL_MIRROR_CONSOLE
#define CSI_CTRL_MIRROR_CONSOLE 1
#endif

/* The UART itself is enabled by defining, in the project settings:
 *     CSI_PRES_UART_BASE    e.g. USART2  (a spare FLEXCOMM routed to header pins;
 *                           pins and clock must also be set up in MCUXpresso Config Tools)
 *     CSI_PRES_UART_FLEXCOMM  FLEXCOMM index of that USART (e.g. 0 for USART0); the
 *                           clock is then read with CLOCK_GetFlexCommClkFreq()
 *     CSI_PRES_UART_CLK_HZ  alternative to CSI_PRES_UART_FLEXCOMM: the clock in Hz
 *     CSI_PRES_UART_BAUD    optional, default 115200 (8N1, no flow control)
 * To use DMA instead of the blocking write, replace uart_tx() in csi_uart_out.c.
 */
#ifndef CSI_PRES_UART_BAUD
#define CSI_PRES_UART_BAUD 115200U
#endif

typedef struct
{
    const char *name;    /* class name, plain [A-Za-z0-9_-] (goes into JSON unescaped) */
    uint8_t cls;         /* class index */
    uint8_t action;      /* CSI_CLASS_ACTION[cls]: 0 hold, 1 ON, 2 OFF */
    uint8_t detected;    /* 1 = occupied */
    uint8_t conf_pct;    /* 0..100: tree-vote share (or vote share if CSI_VOTE_N > 1) */
    int8_t rssi;         /* dBm, mean over the records since the previous result */
    uint32_t latency_us; /* featurize + predict time */
    uint32_t t_ms;       /* uptime when the result was produced */
} csi_result_t;

typedef enum
{
    CSI_ERR_NO_CSI   = 0,
    CSI_ERR_MODEL    = 1, /* one invalid model output: control OFF until the next valid prediction */
    CSI_ERR_SELFTEST = 2, /* boot self-test/init failure: control latched OFF until reboot */
    CSI_ERR_COUNT
} csi_err_t;

/* Sets up the UART (if enabled), the queue and the output task, which sends
 * an OFF control message straight away. 0 = success. The UART's pins and
 * FLEXCOMM clock must already be configured (board code). */
int csi_out_init(void);

/* Re-initialises the control UART peripheral after its state was lost (e.g.
 * PM3 exit), once the board code has restored its pins and clock. 0 = success;
 * no-op when CSI_PRES_UART_BASE is not defined. */
int csi_out_uart_reinit(void);

/* Non-blocking. Returns false (and counts a drop) if the queue is full. */
bool csi_out_post_result(const csi_result_t *r);
void csi_out_post_error(csi_err_t code);

/* Latches the control output OFF ("fault") until reboot. Non-blocking; the
 * latch is also set directly so it survives a full queue. */
void csi_out_post_fault(csi_err_t code);

#endif /* CSI_UART_OUT_H_ */
