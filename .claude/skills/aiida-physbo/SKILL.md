---
name: aiida-physbo
description: "PHYSBO のベイズ最適化を AiiDA の provenance に記録する aiida-physbo を使う・直すときに使う。2 種類の探索空間（候補表 CandidatesData = discrete policy、連続の箱 SearchBoxData = range policy）、ObservationsData の連鎖、calcfunction propose / observe の約束、PhysboOptimizeWorkChain、CLI `physbo-aiida --json` と MCP `physbo-mcp`（ツール physbo_*）、対話ループ（ask–tell）の回し方、事後と獲得関数の取り出し方と図、PHYSBO 3.2.1 の既知の制限（ODAT-SE mapper、range の複数点 + BLM、少数観測でのハイパーパラメータ崩れ）。"
---

# Skill: aiida-physbo（PHYSBO を AiiDA から使う）v1.0.0

このスキルはリポジトリに同梱する公開用です。特定の計算機、アカウント、パスに依る情報は書きません（各自の環境のメモに置く）。詳細は `README.md`、`docs/design.md`、`docs/mcp.md`、`examples/README.md` から辿ります。

## 何が要るか

| もの | 備考 |
|---|---|
| PHYSBO ≥ 3.2 | `pip install -e <PHYSBO>`。range policy と ODAT-SE optimizer は 3 系から。odat-se が依存に入る |
| aiida-core ≥ 2.6 + プロファイル（daemon は WorkChain を使うときだけ） | 対話ループ（propose / observe）は daemon 無しで動く |
| aiida-physbo | `pip install -e ".[mcp,plot,test]"`。entry point（data / calculations / workflows）を変えたら入れ直して `verdi daemon restart` |
| mcp 2.x | MCP を使うときだけ。同じ env の他の MCP サーバと版を揃える |

## 考え方（最初に読む）

- **状態は ObservationsData の連鎖**。`observe` は毎回「これまで全部 + 新規」を持つ新ノードを作り、前のノードに繋ぐ。`propose` は状態を持たず、(空間, 観測) から PHYSBO の Policy を `initial_data` で組み直し、ハイパーパラメータを学習し、`bayes_search(max_num_probes=1, simulator=None)` で次の点だけ返す（PHYSBO の対話モード）。だからどの提案も「どの観測と設定から出たか」を provenance で辿れる。
- **空間は 2 種類**。discrete（`CandidatesData`、配列 `X` (N, d)）では提案は行番号 `actions`（と行 `X`）。range（`SearchBoxData`、`min_X` / `max_X`）では提案は箱の中の座標 `X` で、獲得関数を optimizer（`random`: 一様サンプル、`odatse`: exchange / pamc / minsearch / bayes）で最大化して決める。`propose` は `space` 入力の型で振り分ける。
- **値は生のまま保存し、符号は propose の引数**。PHYSBO は最大化するので既定 `maximize=True`。最小化なら `--minimize`（`maximize=False`）を propose / history / plot に付ける。目的関数値の配列名は PHYSBO に合わせて `t`（いわゆる y）、常に (M, k)。
- **物理的・統計的な判断は固定のツールに任せる**。提案を手で選ばない。LLM が書くのは開発と診断のときだけ。

## 典型的な流れ（CLI。MCP ではツール名が `physbo_<サブコマンド>`、`-` は `_`）

```bash
physbo-aiida --json status                                     # profile / daemon / 版 / ノード数
# 空間
physbo-aiida candidates --file X.csv --columns 0,1 --names a,b      # discrete: 候補表 -> pk S
physbo-aiida candidates --grid '{"min":[-2,-2],"max":[2,2],"num":21}'
physbo-aiida search-box --min -2,-2 --max 2,2 --names x,y           # range: 箱 -> pk S
# ループ
physbo-aiida propose --space-pk S --num-search-each-probe 5 --seed 1          # 観測なし -> random
physbo-aiida observe --space-pk S --actions 12,57 --values 0.3,0.1            # discrete -> pk O1
physbo-aiida observe --space-pk S --x '[[0.3,-1.2],[1.1,0.4]]' --values 0.3,0.1   # range
physbo-aiida propose --space-pk S --observations-pk O1 --score EI --minimize --posterior
physbo-aiida observe --space-pk S --observations-pk O1 --actions 77 --values 1.2   # -> O2（連鎖）
# 読む
physbo-aiida history --pk O2 --minimize        # 観測列、best-so-far / Pareto、observe の連鎖
physbo-aiida proposal --pk <propose pk>        # 提案点と summary（提案点での事後）
physbo-aiida posterior --pk <propose pk>       # --posterior で走らせた手の事後・獲得関数の配列（図のデータ）
physbo-aiida plot --pk O2 --minimize           # best-so-far、2 次元なら観測点、多目的なら Pareto front
physbo-aiida plot --pk <propose pk> --minimize # 1 次元なら事後の帯 + 獲得関数 + 提案点
# 閉ループ（PHYSBO のテスト関数、daemon）
physbo-aiida test-functions
physbo-aiida submit-optimize --test-function Sphere --space range --num-random 10 --num-bayes 20 --score EI
physbo-aiida results --pk <workchain pk>
```

テスト関数は PHYSBO 同梱のもの（Sphere、Rastrigin、Ackley、ZDT、…）に加え、プラグインの `extra_functions.py`（Branin、GoldsteinPrice、SixHumpCamel、Levy、Hartmann3 / 6、Forrester、GramacyLee、DTLZ2）。`test-functions` が既知の最小値を返し、`results` が regret を出す。`--noise σ` で観測ノイズを足せる。`examples/benchmark.py` が一式を回して表と図にする。

獲得関数: 単目的 TS（既定）/ EI / PI、多目的 TS / EHVI / HVPI（値は `--values '[[f1,f2],...]'` の行で渡す）。`num_rand_basis` は 0 で厳密 GP、数百でランダム特徴の BLM（候補が数千以上なら）。

## 約束（破らない）

1. MCP は aiida も physbo も import せず、`physbo-aiida --json` を subprocess で呼ぶだけ。走らせてよい実行体は `physbo-aiida` だけ、サブコマンドは `cli/spec.py` の表だけ。`verdi` は呼べない（ノードを消せない）。
2. `--json` のとき stdout は JSON 1 個（失敗でも `{"ok": false, "error": ...}`）。PHYSBO の表示は stderr。
3. 値は `--flag=value` で渡す（`-4.0,-8.0` のような値が argparse にオプションと読まれないため）。CLI 側も `--flag -4` を `--flag=-4` に書き直す。
4. `--values a,b,c` は「1 目的 × 3 行」。複数目的は `;` 区切りか JSON の行。`--x a,b` は「1 点の座標」。
5. discrete の `observe` は重複 action を拒否する（同じ候補を 2 度測るのは別の設計）。range は同じ座標の再測定を許す（ノイズ）。
6. サブコマンドを足したら `mcp/server.py` に同じ引数名のツールを足す。`test_every_tool_argument_reaches_argv` が見張る。
7. 手法や数値を変えたら版を上げる（`__init__.py` と `pyproject.toml` を一緒に、README の版の表に 1 行）。

## 失敗の見方

- `propose` が `every candidate has been observed already`: discrete で候補を使い切った（WorkChain は exit 420）。
- `['optimizer'] apply to a range space`: discrete 空間に range 専用の引数（optimizer / optimizer_nsamples / odatse_*）を渡した（WorkChain は exit 411）。渡せるのに効かない旗を作らないための拒否。
- `score 'EHVI' is not available for 1 objective(s)`: 目的数と獲得関数の組が違う。`num_objectives` は観測の列数から決まる。
- `stored no posterior`: その手は `--posterior` 無しで走った。事後は提案時にしか作れない（propose は状態を持たない）。
- MCP が `did not finish within 55 s`: 候補が多すぎて厳密 GP が重い。`num_rand_basis` を使う。
- WorkChain が `MissingEntryPointError` で excepted: daemon が aiida-physbo の install 前に起動していた。`verdi daemon restart`。

## PHYSBO 3.2.1 の既知の制限と挙動（プラグインが先に止めるもの）

- ODAT-SE 4 と組むと `mapper` が落ちる（`ColorMap.txt` の先頭行 `fval` を PHYSBO が読み飛ばさない）→ 選択肢から除外。
- range で `num_search_each_probe > 1` かつ `num_rand_basis > 0` は `Variable.add` の形状エラー → 事前に拒否。
- ODAT-SE は作業ディレクトリに `odatse_output/` を書く → プラグインは一時ディレクトリで走らせて消す。
- ODAT-SE の初期点は seed で決まる。propose の `seed` を渡さないと `minsearch` が毎手同じ点を提案し得る（0.3.0 から seed を引き渡す）。
- ベンチマーク（`examples/README.md` の表）: Levy / Hartmann3 は 40 評価で解けるが、値が 3〜10⁶ に広がる Goldstein–Price は素の GP では解けない。`--transform log`（objective の `transform: "log"`）で log f を記録すると discrete では regret 57 → 4.7（range は 1 seed では結論が出ない）。6 次元では獲得関数の最大化に一様サンプル 5000 点のほうが ODAT-SE exchange 300 歩より効いた。ノイズ付きや周期の細かい関数は乱数点を 5〜10 以上に増やす。
- 観測が 3〜6 点のとき、ハイパーパラメータ学習（ML-II）が極端に短い相関長に落ちることがある: 事後平均が平ら、帯が一様、EI が定数（端を提案）。5〜7 点で回復する。プラグインの問題ではない。`examples/figures/` の図に出ている。

## リポジトリの構成

| 場所 | 役割 |
|---|---|
| `aiida_physbo/data.py` | `CandidatesData`、`SearchBoxData`、`ObservationsData`（型付き ArrayData、entry point `physbo.*`） |
| `aiida_physbo/calcfunctions.py` | `candidates_from_file / _grid`、`search_box`、`observe`、`propose`（`PROPOSE_DEFAULTS` が引数の唯一の表）、`evaluate_test_function`、`summarize` |
| `aiida_physbo/workflows/optimize.py` | `PhysboOptimizeWorkChain`（`physbo.optimize`）。random → Bayesian の閉ループ、exit 400 / 410 / 411 / 420 / 421 |
| `aiida_physbo/query/nodes.py` | 読むだけ（status / history / proposal / posterior / results / provenance） |
| `aiida_physbo/cli/spec.py` → `main.py` / `steps.py` / `control.py` | 表 → argparse、ノードを作る側（投入前の検証と行動の記録 `~/.aiida-physbo/log/`）、daemon と kill |
| `aiida_physbo/mcp/server.py` | `physbo-mcp`（`--allow-submit` でノード作成系、`--allow-control` で daemon / kill） |
| `tests/` | 層 1（aiida 不要、spec と MCP の構造）、層 2（`-p aiida.tools.pytest_fixtures` で一時 sqlite プロファイル） |
| `examples/` | CLI の対話ループと閉ループの script、1 次元の図（`one_dimensional.py`）、MCP セッションの記録 |

## 未対応

- 目的関数が AiiDA のプロセス（CalcJob）であるループ。対話モードで代替する（propose → 投入 → observe）。
- 3 次元以上の箱での事後の格子（`posterior` は dim ≤ 2）。
