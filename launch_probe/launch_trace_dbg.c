/* launch_trace.c —— 拦截 dlsym()/cuGetProcAddress()，把 CUDA driver 的
 * 装载与发射函数换成自己的包装，从而观察 cuDNN 引擎库如何唤起 kernel。
 *
 *   LD_PRELOAD=launch_trace.so ./demo
 *
 * 记录内容：cuModuleLoadData(镜像 magic)、cuModuleGetFunction(函数名)、
 *          cuFuncSetAttribute(含 dynamic smem 上限)、cuLaunchKernel(_ptsz)、
 *          cuLaunchKernelEx(_ptsz)（grid/block/smem/stream/attrs/参数）。
 * 参数值读取带 SIGSEGV/SIGBUS 保护（读不到就打 ?）。
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <signal.h>
#include <setjmp.h>
#include <execinfo.h>
#include <stdlib.h>

typedef int CUresult;
typedef void *CUfunction, *CUmodule, *CUstream, *CUlibrary, *CUkernel;

typedef struct {
    unsigned gx, gy, gz, bx, by, bz, smem;
    CUstream stream;
    void *attrs;
    unsigned numAttrs;
} launchcfg_t;

typedef struct { int id; char p[4]; unsigned char v[64]; } launchattr_t;

/* ---------- 真实函数指针 ---------- */
static void *(*real_dlsym)(void *, const char *);
static CUresult (*r_moduleLoadData)(CUmodule *, const void *);
static CUresult (*r_moduleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*r_funcSetAttribute)(CUfunction, int, int);
static CUresult (*r_launchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                  unsigned, unsigned, unsigned, CUstream, void **, void **);
static CUresult (*r_launchKernelEx)(const launchcfg_t *, CUfunction, void **, void **);
static CUresult (*r_getProcAddress)(const char *, void **, int, unsigned long long);
static CUresult (*r_getProcAddress_v2)(const char *, void **, int, unsigned long long, void *);

/* ---------- 函数名表 ---------- */
#define MAXFN 8192
static struct { CUfunction f; char name[192]; } g_fns[MAXFN];
static int g_nfns;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *fn_name(CUfunction f) {
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_nfns; i++)
        if (g_fns[i].f == f) { const char *n = g_fns[i].name; pthread_mutex_unlock(&g_lock); return n; }
    pthread_mutex_unlock(&g_lock);
    return "?";
}
static void fn_add(CUfunction f, const char *n) {
    pthread_mutex_lock(&g_lock);
    if (g_nfns < MAXFN) { g_fns[g_nfns].f = f; snprintf(g_fns[g_nfns].name, sizeof g_fns[0].name, "%s", n); g_nfns++; }
    pthread_mutex_unlock(&g_lock);
}

/* ---------- 受保护的读 ---------- */
static __thread sigjmp_buf g_jb;
static __thread volatile int g_in_safe;
static struct sigaction old_segv, old_bus;
static int handler_ready;

static void fault_handler(int sig, siginfo_t *si, void *ctx) {
    (void)si; (void)ctx;
    if (g_in_safe) siglongjmp(g_jb, 1);
    void *bt[32];
    int n = backtrace(bt, 32);
    fprintf(stderr, "[fault] signal %d backtrace:\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(139);
}
static void install_handlers(void) {
    if (handler_ready) return;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);
    handler_ready = 1;
}
static int safe_read(const void *p, void *dst, size_t n) {
    install_handlers();
    g_in_safe = 1;
    if (sigsetjmp(g_jb, 1) == 0) { memcpy(dst, p, n); g_in_safe = 0; return 1; }
    g_in_safe = 0;
    return 0;
}

static const char *caller(void) {
    Dl_info info;
    if (dladdr(__builtin_return_address(0), &info) && info.dli_fname) {
        const char *b = strrchr(info.dli_fname, '/');
        return b ? b + 1 : info.dli_fname;
    }
    return "?";
}

/* ---------- 包装 ---------- */
static CUresult w_moduleLoadData(CUmodule *m, const void *img) {
    fprintf(stderr, "[dbg] w_moduleLoadData in\n");
    CUresult rc = r_moduleLoadData(m, img);
    unsigned char h[4] = {0}; char kind[24] = "?";
    if (rc == 0 && img && safe_read(img, h, 4)) {
        uint32_t v; memcpy(&v, h, 4);
        if (v == 0x464c457f) snprintf(kind, sizeof kind, "ELF(cubin/IR)");
        else if (v == 0xba55ed50) snprintf(kind, sizeof kind, "fatbin");
        else if (v == 0x466243b1) snprintf(kind, sizeof kind, "cubin");
        else snprintf(kind, sizeof kind, "magic=%08x", v);
    }
    flockfile(stderr);
    fprintf(stderr, "[img ] <- %-34s cuModuleLoadData  module=%p rc=%d kind=%s\n", caller(), (void *)m, rc, kind);
    funlockfile(stderr);
    return rc;
}

static CUresult w_moduleGetFunction(CUfunction *f, CUmodule m, const char *name) {
    fprintf(stderr, "[dbg] w_moduleGetFunction in\n");
    CUresult rc = r_moduleGetFunction(f, m, name);
    if (rc == 0 && f) fn_add(*f, name ? name : "?");
    flockfile(stderr);
    fprintf(stderr, "[func] <- %-34s cuModuleGetFunction  fn=%p rc=%d name=%s\n", caller(), f ? (void *)*f : NULL, rc, name ? name : "?");
    funlockfile(stderr);
    return rc;
}

static const char *attr_name(int a) {
    switch (a) {
        case 0: return "MAX_THREADS_PER_BLOCK"; case 1: return "SHARED_SIZE_BYTES";
        case 2: return "CONST_SIZE_BYTES";      case 3: return "LOCAL_SIZE_BYTES";
        case 4: return "NUM_REGS";              case 5: return "PTX_VERSION";
        case 6: return "BINARY_VERSION";        case 7: return "CACHE_MODE_CA";
        case 8: return "MAX_DYNAMIC_SHARED_SIZE_BYTES";
        case 9: return "PREFERRED_SHARED_MEMORY_CARVEOUT";
        case 10: return "CLUSTER_SIZE";         default: return "?";
    }
}
static CUresult w_funcSetAttribute(CUfunction f, int attr, int val) {
    CUresult rc = r_funcSetAttribute(f, attr, val);
    flockfile(stderr);
    fprintf(stderr, "[attr] <- %-34s cuFuncSetAttribute  fn=%s attr=%s(%d) val=%d rc=%d\n", caller(), fn_name(f), attr_name(attr), attr, val, rc);
    funlockfile(stderr);
    return rc;
}

static void log_params(void **kp, void **extra) {
    flockfile(stderr);
    if (kp) {
        fprintf(stderr, "  params:");
        for (int i = 0; i < 12; i++) {
            void *p = NULL;
            if (!safe_read(kp + i, &p, sizeof p)) { fprintf(stderr, " [out-of-range]"); break; }
            if (!p) { fprintf(stderr, " null"); continue; }
            unsigned char buf[16] = {0};
            if (safe_read(p, buf, sizeof buf)) {
                uint64_t a, b;
                memcpy(&a, buf, 8); memcpy(&b, buf + 8, 8);
                fprintf(stderr, " p%d(%p)=%016llx,%016llx", i, p, (unsigned long long)a, (unsigned long long)b);
            } else {
                fprintf(stderr, " p%d(%p)=?", i, p);
            }
        }
        fprintf(stderr, "\n");
    }
    if (extra) fprintf(stderr, "  extra=%p\n", (void *)extra);
    funlockfile(stderr);
}

static CUresult w_launchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz, unsigned smem,
                               CUstream st, void **kp, void **extra) {
    fprintf(stderr, "[dbg] w_launchKernel in\n");
    flockfile(stderr);
    fprintf(stderr, "[launch] <- %-34s cuLaunchKernel  kernel=%s\n  grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u stream=%p\n",
            caller(), fn_name(f), gx, gy, gz, bx, by, bz, smem, st);
    funlockfile(stderr);
    log_params(kp, extra);
    return r_launchKernel(f, gx, gy, gz, bx, by, bz, smem, st, kp, extra);
}

static CUresult w_launchKernelEx(const launchcfg_t *cfg, CUfunction f, void **kp, void **extra) {
    flockfile(stderr);
    fprintf(stderr, "[launch] <- %-34s cuLaunchKernelEx  kernel=%s\n  grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u stream=%p attrs=%u\n",
            caller(), fn_name(f), cfg->gx, cfg->gy, cfg->gz, cfg->bx, cfg->by, cfg->bz,
            cfg->smem, cfg->stream, cfg->numAttrs);
    if (cfg->attrs && cfg->numAttrs) {
        const launchattr_t *a = (const launchattr_t *)cfg->attrs;
        fprintf(stderr, "  attr-ids:");
        for (unsigned i = 0; i < cfg->numAttrs && i < 8; i++) fprintf(stderr, " %d", a[i].id);
        fprintf(stderr, "\n");
    }
    funlockfile(stderr);
    log_params(kp, extra);
    return r_launchKernelEx(cfg, f, kp, extra);
}

/* ---------- cuGetProcAddress 递归保护（WSL 版 libcuda 会再取一次） ---------- */
static __thread int g_in_gpa;

/* ---------- dlsym 钩子：把真实指针换成包装 ---------- */
static void *pick(const char *name, void *real) {
    if (!strcmp(name, "cuModuleLoadData"))      { r_moduleLoadData = real;    return (void *)w_moduleLoadData; }
    if (!strcmp(name, "cuModuleGetFunction"))   { r_moduleGetFunction = real; return (void *)w_moduleGetFunction; }
    if (!strcmp(name, "cuFuncSetAttribute"))    { r_funcSetAttribute = real;  return (void *)w_funcSetAttribute; }
    if (!strcmp(name, "cuLaunchKernel") || !strcmp(name, "cuLaunchKernel_ptsz"))
                                                { r_launchKernel = real;      return (void *)w_launchKernel; }
    if (!strcmp(name, "cuLaunchKernelEx") || !strcmp(name, "cuLaunchKernelEx_ptsz"))
                                                { r_launchKernelEx = real;    return (void *)w_launchKernelEx; }
    if (!strcmp(name, "cuGetProcAddress"))      { r_getProcAddress = real;    return (void *)0; } /* 见下 */
    if (!strcmp(name, "cuGetProcAddress_v2"))   { r_getProcAddress_v2 = real; return (void *)0; }
    return real;
}

static CUresult w_getProcAddress_v2(const char *sym, void **pfn, int ver, unsigned long long flags, void *status);
static CUresult w_getProcAddress(const char *sym, void **pfn, int ver, unsigned long long flags);

static CUresult w_getProcAddress_v2(const char *sym, void **pfn, int ver, unsigned long long flags, void *status) {
    fprintf(stderr, "[dbg] w_gpa2 in sym=%s r=%p\n", sym ? sym : "(null)", (void *)r_getProcAddress_v2);
    if (!r_getProcAddress_v2) return -1;
    CUresult rc = r_getProcAddress_v2(sym, pfn, ver, flags, status);
    fprintf(stderr, "[dbg] w_gpa2 rc=%d pfn=%p\n", rc, pfn ? *pfn : 0);
    if (rc == 0 && pfn && *pfn && sym) { void *w = pick(sym, *pfn); if (w) *pfn = w; }
    return rc;
}
static CUresult w_getProcAddress(const char *sym, void **pfn, int ver, unsigned long long flags) {
    if (!r_getProcAddress) return -1;
    if (g_in_gpa) return r_getProcAddress(sym, pfn, ver, flags);
    g_in_gpa = 1;
    CUresult rc = r_getProcAddress(sym, pfn, ver, flags);
    g_in_gpa = 0;
    if (rc == 0 && pfn && *pfn) { void *w = pick(sym, *pfn); if (w) *pfn = w; }
    return rc;
}

void *dlsym(void *handle, const char *name) {
    if (name && name[0]=='c' && name[1]=='u' && name[2]>='A' && name[2]<='Z')
        fprintf(stderr, "[dbg] dlsym(%s) enter\n", name);
    if (!real_dlsym) real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym) { fprintf(stderr, "[dbg] real_dlsym NULL for %s\n", name?name:"(null)"); return 0; }
    void *real = real_dlsym(handle, name);
    if (name && name[0]=='c' && name[1]=='u' && name[2]>='A' && name[2]<='Z')
        fprintf(stderr, "[dbg] dlsym(%s) = %p\n", name, real);
    if (!real || !name) return real;
    /* 不包装 cuGetProcAddress*：libcuda 内部会再次经 PLT 调到本符号，
       形成 ping-pong 环路；引擎库实际是用 dlsym 取 cuLaunchKernel 的。 */
    return pick(name, real);
}
