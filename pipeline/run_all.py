"""
Orquestrador do Pipeline de Dados — Projeto WordGen
Executa em ordem:
  1. pipeline_datasets.py
  2. fix_escolas.py
  3. finalizar_datasets.py

Para no primeiro erro e imprime o tempo total.
"""
import subprocess
import sys
import time
from pathlib import Path


def main():
    inicio_total = time.time()
    pipeline_dir = Path(__file__).resolve().parent

    scripts = [
        "pipeline_datasets.py",
        "fix_escolas.py",
        "finalizar_datasets.py",
    ]

    for script_name in scripts:
        script_path = pipeline_dir / script_name
        print(f"\n{'=' * 60}")
        print(f"Executando {script_name}...")
        print(f"{'=' * 60}\n")
        inicio_etapa = time.time()
        res = subprocess.run([sys.executable, str(script_path)], check=False)
        duracao_etapa = time.time() - inicio_etapa
        if res.returncode != 0:
            print(
                f"\n[ERRO] Falha na execução de {script_name} (código de saída: {res.returncode})",
                file=sys.stderr,
            )
            sys.exit(res.returncode)
        print(f"\n[OK] {script_name} finalizado em {duracao_etapa:.2f}s")

    duracao_total = time.time() - inicio_total
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETO CONCLUÍDO COM SUCESSO EM {duracao_total:.2f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
