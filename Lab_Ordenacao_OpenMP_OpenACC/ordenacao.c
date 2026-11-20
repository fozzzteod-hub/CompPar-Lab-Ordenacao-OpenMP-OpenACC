/* Rafael Teodoro Cruz - RA: 10723258 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <openacc.h>

static void *alocar(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "Falha de memoria\n"); exit(1); }
    return p;
}

static int comparar(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* Heapsort local executavel no dispositivo, sem qsort no kernel. */
#pragma acc routine seq
static void descer(int *a, int n, int raiz) {
    while (raiz < n / 2) {
        int filho = 2 * raiz + 1;
        if (filho + 1 < n && a[filho] < a[filho + 1]) filho++;
        if (a[raiz] >= a[filho]) break;
        int t = a[raiz]; a[raiz] = a[filho]; a[filho] = t;
        raiz = filho;
    }
}

#pragma acc routine seq
static void ordenar_local(int *a, int n) {
    for (int i = n / 2; i > 0; i--) descer(a, n, i - 1);
    for (int i = n - 1; i > 0; i--) {
        int t = a[0]; a[0] = a[i]; a[i] = t;
        descer(a, i, 0);
    }
}

/* Mescla os dois blocos originais antes de alterar qualquer um deles. */
#pragma acc routine seq
static void compare_split(int *a, int *tmp, int m) {
    int i = 0, j = m;
    for (int k = 0; k < 2*m; k++) {
        if (j == 2*m || (i < m && a[i] <= a[j])) tmp[k] = a[i++];
        else tmp[k] = a[j++];
    }
    for (int k = 0; k < 2*m; k++) a[k] = tmp[k];
}

void odd_even_sort_omp(int *data, int n, int p) {
    int m = n / p;
    int *tmp = alocar((size_t)n * sizeof(int));
    #pragma omp parallel num_threads(p)
    {
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++)
            qsort(data + b*m, m, sizeof(int), comparar);
        for (int fase = 0; fase < p; fase++) {
            int inicio = fase % 2;
            #pragma omp for schedule(static)
            for (int b = inicio; b < p-1; b += 2)
                compare_split(data + b*m, tmp + b*m, m);
            /* barreira implicita do omp for ao final da fase */
        }
    }
    free(tmp);
}

void odd_even_sort_acc(int *data, int n, int p) {
    int m = n / p;
    int *tmp = alocar((size_t)n * sizeof(int));
    #pragma acc data copy(data[0:n]) create(tmp[0:n])
    {
        #pragma acc parallel loop gang present(data[0:n])
        for (int b = 0; b < p; b++) ordenar_local(data + b*m, m);
        for (int fase = 0; fase < p; fase++) {
            int inicio = fase % 2;
            #pragma acc parallel loop gang present(data[0:n],tmp[0:n])
            for (int b = inicio; b < p-1; b += 2)
                compare_split(data + b*m, tmp + b*m, m);
            /* kernels sincronos: a fase seguinte aguarda a anterior */
        }
    }
    free(tmp);
}

/* Cada bloco ja ordenado seleciona p-1 amostras regulares. */
#pragma acc routine seq
static void amostrar(const int *a, int *s, int m, int p) {
    for (int j = 1; j < p; j++) s[j-1] = a[(long long)j*m/p];
}

static void escolher_splitters(int *amostras, int *splitters, int p) {
    int ns = p * (p-1);
    if (p == 1) return;
    qsort(amostras, ns, sizeof(int), comparar);
    for (int j = 1; j < p; j++) splitters[j-1] = amostras[j*(p-1)];
}

/* Particiona cada bloco ordenado: fronteiras dos p baldes. */
#pragma acc routine seq
static void particionar(const int *a, int m, const int *splitters, int p, int *lim) {
    int k = 0;
    lim[0] = 0;
    for (int b = 0; b < p-1; b++) {
        while (k < m && a[k] <= splitters[b]) k++;
        lim[b+1] = k;
    }
    lim[p] = m;
}

/* Offset de cada origem dentro de um destino; os segmentos nao se sobrepoem. */
#pragma acc routine seq
static void offsets_destino(const int *lim, int *off, int *totais, int p, int b) {
    int pos = 0;
    for (int origem = 0; origem < p; origem++) {
        off[origem*p+b] = pos;
        pos += lim[origem*(p+1)+b+1] - lim[origem*(p+1)+b];
    }
    totais[b] = pos;
}

#pragma acc routine seq
static void prefixos(const int *totais, int *base, int p) {
    base[0] = 0;
    for (int b = 0; b < p; b++) base[b+1] = base[b] + totais[b];
}

#pragma acc routine seq
static void redistribuir(const int *a, int *saida, int m, int p, int origem,
                        const int *lim, const int *off, const int *base) {
    for (int b = 0; b < p; b++) {
        int ini = lim[origem*(p+1)+b];
        int fim = lim[origem*(p+1)+b+1];
        int dest = base[b] + off[origem*p+b];
        for (int k = ini; k < fim; k++) saida[dest+k-ini] = a[origem*m+k];
    }
}

void sample_sort_omp(int *data, int n, int p) {
    int m = n/p, ns = p*(p-1), nl = p*(p+1), no = p*p;
    if (ns == 0) ns = 1;
    int *samples = alocar((size_t)ns*sizeof(int));
    int *split = alocar((size_t)p*sizeof(int));
    int *lim = alocar((size_t)nl*sizeof(int));
    int *off = alocar((size_t)no*sizeof(int));
    int *tot = alocar((size_t)p*sizeof(int));
    int *base = alocar((size_t)(p+1)*sizeof(int));
    int *out = alocar((size_t)n*sizeof(int));
    #pragma omp parallel num_threads(p)
    {
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++) {
            qsort(data+b*m, m, sizeof(int), comparar);
            amostrar(data+b*m, samples+b*(p-1), m, p);
        }
        #pragma omp single
        escolher_splitters(samples, split, p);
        /* single: gather, escolha centralizada e broadcast por memoria compartilhada */
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++) particionar(data+b*m,m,split,p,lim+b*(p+1));
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++) offsets_destino(lim,off,tot,p,b);
        #pragma omp single
        prefixos(tot,base,p);
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++) redistribuir(data,out,m,p,b,lim,off,base);
        #pragma omp for schedule(static)
        for (int b = 0; b < p; b++) qsort(out+base[b],tot[b],sizeof(int),comparar);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) data[i] = out[i];
    }
    free(samples); free(split); free(lim); free(off); free(tot); free(base); free(out);
}

void sample_sort_acc(int *data, int n, int p) {
    int m = n/p, ns = p*(p-1), nl = p*(p+1), no = p*p;
    if (ns == 0) ns = 1;
    int *samples = alocar((size_t)ns*sizeof(int));
    int *split = alocar((size_t)p*sizeof(int));
    int *lim = alocar((size_t)nl*sizeof(int));
    int *off = alocar((size_t)no*sizeof(int));
    int *tot = alocar((size_t)p*sizeof(int));
    int *base = alocar((size_t)(p+1)*sizeof(int));
    int *out = alocar((size_t)n*sizeof(int));
    #pragma acc data copy(data[0:n]) create(samples[0:ns],split[0:p],lim[0:nl],off[0:no],tot[0:p],base[0:p+1],out[0:n])
    {
        #pragma acc parallel loop gang present(data[0:n],samples[0:ns])
        for (int b = 0; b < p; b++) {
            ordenar_local(data+b*m,m);
            amostrar(data+b*m,samples+b*(p-1),m,p);
        }
        if (p > 1) {
            #pragma acc update self(samples[0:ns])
            escolher_splitters(samples,split,p);
            #pragma acc update device(split[0:p-1])
        }
        #pragma acc parallel loop gang present(data[0:n],split[0:p],lim[0:nl])
        for (int b = 0; b < p; b++) particionar(data+b*m,m,split,p,lim+b*(p+1));
        #pragma acc parallel loop gang present(lim[0:nl],off[0:no],tot[0:p])
        for (int b = 0; b < p; b++) offsets_destino(lim,off,tot,p,b);
        #pragma acc serial present(tot[0:p],base[0:p+1])
        { prefixos(tot,base,p); }
        #pragma acc parallel loop gang present(data[0:n],out[0:n],lim[0:nl],off[0:no],base[0:p+1])
        for (int b = 0; b < p; b++) redistribuir(data,out,m,p,b,lim,off,base);
        #pragma acc parallel loop gang present(out[0:n],base[0:p+1],tot[0:p])
        for (int b = 0; b < p; b++) ordenar_local(out+base[b],tot[b]);
        #pragma acc parallel loop present(data[0:n],out[0:n])
        for (int i = 0; i < n; i++) data[i] = out[i];
    }
    free(samples); free(split); free(lim); free(off); free(tot); free(base); free(out);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr,"Uso: %s odd|sample serial|omp|acc n p [semente] [--host-test]\n", argv[0]);
        return 1;
    }
    const char *alg=argv[1], *modo=argv[2];
    int n=atoi(argv[3]), p=atoi(argv[4]);
    unsigned seed=argc>=6?(unsigned)strtoul(argv[5],NULL,10):42;
    int host=argc==7 && strcmp(argv[6],"--host-test")==0;
    if (n<=0 || p<=0 || p>1024 || n%p || (strcmp(alg,"odd")&&strcmp(alg,"sample")) || (strcmp(modo,"serial")&&strcmp(modo,"omp")&&strcmp(modo,"acc"))) {
        fprintf(stderr,"Parametros invalidos: n positivo e multiplo de p, 1 <= p <= 1024.\n"); return 1;
    }
    int *data=alocar((size_t)n*sizeof(int)), *ref=alocar((size_t)n*sizeof(int));
    srand(seed);
    for (int i=0;i<n;i++) data[i]=rand()%(n/2+1);
    memcpy(ref,data,(size_t)n*sizeof(int));
    int omp_threads=1;
    omp_set_dynamic(0);
    omp_set_num_threads(p);
    if (!strcmp(modo,"omp")) {
        #pragma omp parallel
        {
            #pragma omp single
            omp_threads=omp_get_num_threads();
        }
        if (omp_threads!=p) { fprintf(stderr,"Equipe real: %d; solicitada: %d\n",omp_threads,p); return 1; }
    }
    if (!strcmp(modo,"acc")) {
        if (host) acc_init(acc_device_host);
        else {
            if (acc_get_num_devices(acc_device_nvidia)<1) {fprintf(stderr,"Nenhuma GPU NVIDIA disponivel; benchmark GPU cancelado.\n"); return 2;}
            acc_init(acc_device_nvidia);
        }
    }
    clock_t cpu0=clock();
    double t0=omp_get_wtime();
    if (!strcmp(modo,"serial")) qsort(data,n,sizeof(int),comparar);
    else if (!strcmp(alg,"odd")) {
        if (!strcmp(modo,"omp")) odd_even_sort_omp(data,n,p);
        else odd_even_sort_acc(data,n,p);
    } else {
        if (!strcmp(modo,"omp")) sample_sort_omp(data,n,p);
        else sample_sort_acc(data,n,p);
    }
    double elapsed=omp_get_wtime()-t0;
    double cpu=(double)(clock()-cpu0)/CLOCKS_PER_SEC;
    qsort(ref,n,sizeof(int),comparar);
    int ok=memcmp(data,ref,(size_t)n*sizeof(int))==0;
    printf("%s,%s,%d,%d,%u,%.9f,%.9f,%s,%s\n",alg,modo,n,p,seed,elapsed,cpu,ok?"OK":"ERRO",!strcmp(modo,"acc")?(host?"HOST_TEST":"NVIDIA"):"CPU");
    free(data);free(ref);
    return ok?0:3;
}
