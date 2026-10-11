/* Read-only observation of Z3 check results; no parameters or results changed. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
static void *symbol(const char *name) {
    void *h = dlopen(getenv("TASK_C_Z3_LIB"), RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "observer dlopen failed: %s\n", dlerror()); abort(); }
    void *p = dlsym(h, name);
    if (!p) abort();
    return p;
}
static void record(void *ctx, void *obj, int result, int optimize) {
    const char *reason = "";
    if (!result) {
        const char *(*get_reason)(void *, void *) = symbol(optimize ? "Z3_optimize_get_reason_unknown" : "Z3_solver_get_reason_unknown");
        reason = get_reason(ctx, obj);
    }
    fprintf(stderr, "[task-c-z3] result=%d reason=%s\n", result, reason ? reason : "");
}
int Z3_solver_check(void *ctx, void *solver) {
    int (*real)(void *, void *) = symbol("Z3_solver_check");
    int result = real(ctx, solver); record(ctx, solver, result, 0); return result;
}
int Z3_solver_check_assumptions(void *ctx, void *solver, unsigned n, void *const *args) {
    int (*real)(void *, void *, unsigned, void *const *) = symbol("Z3_solver_check_assumptions");
    int result = real(ctx, solver, n, args); record(ctx, solver, result, 0); return result;
}
int Z3_optimize_check(void *ctx, void *opt, unsigned n, void *const *args) {
    int (*real)(void *, void *, unsigned, void *const *) = symbol("Z3_optimize_check");
    int result = real(ctx, opt, n, args); record(ctx, opt, result, 1); return result;
}
