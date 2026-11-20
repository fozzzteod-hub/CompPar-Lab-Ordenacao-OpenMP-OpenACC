import csv
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

# Ajuste se necessario. n deve ser divisivel por p.
TAMANHOS = [8192, 32768, 131072]
BLOCOS = [1, 2, 4, 8, 16]
REPETICOES = 3
SEMENTE = 42

Path('resultados').mkdir(exist_ok=True)
campos = ['algoritmo', 'versao', 'n', 'p', 'semente', 'tempo_s', 'tempo_cpu_s', 'verificacao', 'dispositivo']
registros = defaultdict(list)
with open('resultados/tempos.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(campos)
    for alg in ['odd', 'sample']:
        for n in TAMANHOS:
            for p in BLOCOS:
                for r in range(REPETICOES):
                    for modo in ['serial', 'omp', 'acc']:
                        cmd = ['./ordenacao', alg, modo, str(n), str(p), str(SEMENTE)]
                        linha = subprocess.check_output(cmd, text=True).strip()
                        dados = next(csv.reader([linha]))
                        if len(dados) != 9 or dados[7] != 'OK' or (modo == 'acc' and dados[8] != 'NVIDIA'):
                            raise RuntimeError('Execucao invalida: ' + linha)
                        w.writerow(dados)
                        f.flush()
                        registros[(alg, n, p, modo)].append(float(dados[5]))
                        print(linha, flush=True)

with open('resultados/medias.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['algoritmo','n','p','serial_s','openmp_s','openacc_s','speedup_omp','eficiencia_omp_pct','speedup_acc','eficiencia_acc_logica_pct'])
    for alg in ['odd', 'sample']:
        for n in TAMANHOS:
            for p in BLOCOS:
                ts,to,ta = [statistics.mean(registros[alg,n,p,m]) for m in ['serial','omp','acc']]
                so,sa = ts/to, ts/ta
                w.writerow([alg,n,p,*(f'{x:.9f}' for x in (ts,to,ta)),f'{so:.6f}',f'{100*so/p:.4f}',f'{sa:.6f}',f'{100*sa/p:.4f}'])
print('Arquivos gravados em resultados/tempos.csv e resultados/medias.csv')
