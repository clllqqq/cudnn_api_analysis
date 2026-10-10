#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

static void *(*real_dlsym)(void *, const char *);
static __thread int in_hook;

void *dlsym(void *handle, const char *name) {
    if (!real_dlsym)
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!in_hook && name && name[0]=='c' && name[1]=='u' && name[2] >= 'A' && name[2] <= 'Z') {
        in_hook = 1;
        Dl_info info;
        if (dladdr(__builtin_return_address(0), &info) && info.dli_fname)
            fprintf(stderr, "[dlsym] %-28s <- %s\n", name, info.dli_fname);
        in_hook = 0;
    }
    return real_dlsym(handle, name);
}
