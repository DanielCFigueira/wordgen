"""
Utilitário para construção e execução de Jupyter Notebooks com captura de saídas.
Permite gerar notebooks 100% executados com tabelas, gráficos matplotlib e markdown.
"""

import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path.home() / "AppData" / "Local" / "Temp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class NotebookBuilder:
    def __init__(self, title: str = "Notebook"):
        self.title = title
        self.cells = []
        self.execution_count = 0
        self.namespace = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
        self.current_cell_outputs = []

        def custom_display(*args, **kwargs):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    self.current_cell_outputs.append(
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/html": [arg.to_html()],
                                "text/plain": [arg.to_string()],
                            },
                            "metadata": {},
                        }
                    )
                elif isinstance(arg, dict):
                    self.current_cell_outputs.append(
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/plain": [json.dumps(arg, indent=2, ensure_ascii=False)],
                            },
                            "metadata": {},
                        }
                    )
                else:
                    self.current_cell_outputs.append(
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/plain": [str(arg)],
                            },
                            "metadata": {},
                        }
                    )

        self.namespace["display"] = custom_display

    def add_markdown(self, source: str):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        self.cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            }
        )

    def add_code(self, code_source: str, execute: bool = True):
        self.execution_count += 1
        lines = [line + "\n" for line in code_source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")

        outputs = []
        if execute:
            self.current_cell_outputs = []
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf

            try:
                exec(code_source, self.namespace)
            except Exception as e:
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                outputs.append(
                    {
                        "output_type": "error",
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": tb_lines,
                    }
                )
                print(f"[ERRO NA CÉLULA {self.execution_count}]: {e}", file=old_stderr)
                traceback.print_exc(file=old_stderr)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            stdout_text = stdout_buf.getvalue()
            if stdout_text:
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [line + "\n" for line in stdout_text.splitlines(keepends=False)],
                    }
                )

            stderr_text = stderr_buf.getvalue()
            if stderr_text:
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": [line + "\n" for line in stderr_text.splitlines(keepends=False)],
                    }
                )

            outputs.extend(self.current_cell_outputs)

            fig_nums = plt.get_fignums()
            for num in fig_nums:
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                outputs.append(
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": img_b64,
                            "text_plain": [
                                f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"
                            ],
                        },
                        "metadata": {},
                    }
                )
            plt.close("all")

        self.cells.append(
            {
                "cell_type": "code",
                "execution_count": self.execution_count,
                "metadata": {},
                "source": lines,
                "outputs": outputs,
            }
        )

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nb_json = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3 (ipykernel)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.12.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=1, ensure_ascii=False)
        print(f"[NOTEBOOK SALVO]: {filepath} ({len(self.cells)} células)")
