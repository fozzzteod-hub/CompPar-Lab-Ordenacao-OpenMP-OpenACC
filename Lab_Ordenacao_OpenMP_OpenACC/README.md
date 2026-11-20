# CompPar — Algoritmos de Ordenação com OpenMP e OpenACC

Aluno: Rafael Teodoro Cruz — RA 10723258

Atividade individual. Implementações de Odd-Even Transposition Sort (blocos) e Sample Sort, com referência serial qsort. Os quatro nomes de função do enunciado estão em `ordenacao.c`.

## Ambiente e compilação

Use preferencialmente uma única sessão do Google Colab com GPU NVIDIA para comparar CPU e GPU. Instale NVIDIA HPC SDK (nvc), conforme o notebook fornecido pelo professor; verifique a versão disponível antes de instalar. Em um terminal bash/uma célula `%%bash`:

```bash
nvidia-smi
lscpu
free -h
NVC=$(find /opt/nvidia/hpc_sdk -path '*/compilers/bin/nvc' -type f 2>/dev/null | sort -V | tail -1)
test -n "$NVC" || { echo 'Instale o NVIDIA HPC SDK'; exit 1; }
"$NVC" -V
"$NVC" -O3 -mp -acc -Minfo=accel ordenacao.c -o ordenacao
```

## Verificação

```bash
./ordenacao odd serial 8192 4
./ordenacao odd omp 8192 4
./ordenacao odd acc 8192 4
./ordenacao sample serial 8192 4
./ordenacao sample omp 8192 4
./ordenacao sample acc 8192 4
```

Cada linha indica algoritmo, versão, n, p, semente, tempo de parede em segundos, tempo de CPU em segundos, verificação e dispositivo. Os dados aleatórios são idênticos por semente e tamanho. O resultado é comparado elemento a elemento com qsort, inclusive duplicatas. O programa recusa benchmark OpenACC sem GPU NVIDIA.

Para teste funcional local sem GPU (não usar os tempos para conclusão sobre GPU):

```bash
gcc -O3 -Wall -Wextra -fopenmp -fopenacc ordenacao.c -o ordenacao
./ordenacao odd acc 8192 4 42 --host-test
./ordenacao sample acc 8192 4 42 --host-test
```

## Benchmark

```bash
python3 benchmark.py
```

Gera `resultados/tempos.csv` (execuções individuais) e `resultados/medias.csv` (médias, speedup e eficiência). São usados n=8192,32768,131072; p=1,2,4,8 e três repetições. O script interrompe diante de erros; não registra fallback CPU como GPU. A região medida inclui alocação auxiliar, fases do algoritmo e transferências OpenACC; exclui geração de entrada, validação e inicialização prévia do contexto OpenACC. Usa tempo de parede comum via omp_get_wtime para a comparação e registra clock() separadamente, especialmente para a referência serial. Não usar tempo de CPU para comparar CPU multithread com GPU.

## Estratégias

- Odd-Even: ordenação local e p fases pares/ímpares; cada par mescla os dois blocos antes de modificar os originais, preservando a multiplicidade. Barreiras implícitas em OpenMP, kernels síncronos em OpenACC.
- Sample Sort: amostras regulares, coleta e seleção central de splitters, distribuição dos splitters, partição local, cálculo de offsets exclusivos, redistribuição all-to-all e ordenação local final. Na GPU, apenas as amostras e splitters são transferidos para a etapa central; as ordenações locais, particionamento, offsets, redistribuição e ordenação final executam no dispositivo.
- OpenACC: heapsort local sequencial por bloco (rotina de dispositivo), com blocos independentes distribuídos em gangs. É uma estratégia simples, não uma ordenação interna de cada bloco plenamente vetorizada. Pode subutilizar a GPU para p pequeno e para partições desbalanceadas. Duplicatas podem concentrar valores em um balde.
- OpenMP: p threads solicitadas; os loops de blocos/p pares podem deixar threads ociosas. As regiões de saída dos participantes são disjuntas, não requerendo critical durante a redistribuição.
- Eficiência OpenMP = S/p. Eficiência OpenACC = S/p como indicador por bloco lógico, conforme parâmetro do enunciado; p não representa o número real de threads ou núcleos da GPU. Não interpretar esse valor como utilização física da GPU.
- A análise final deve considerar CPUs disponíveis, GPU utilizada, granularidade, estabilidade das medições, overhead, comunicação e a complexidade teórica do material. O relatório será escrito com os tempos reais da execução GPU e CPU; não há resultados de GPU inventados neste pacote.
