#ifndef EVAL_FUNC_H
#define EVAL_FUNC_H

#ifdef __cplusplus
extern "C" {
#endif

void set_func(int funcID);
void set_data_dir(char *new_data_dir);
double eval_sol(double *x);
void free_func(void);
void next_run(void);

#ifdef __cplusplus
}
#endif

#endif // EVAL_FUNC_H


