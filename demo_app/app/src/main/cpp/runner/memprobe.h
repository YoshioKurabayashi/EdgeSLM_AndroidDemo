// EdgeSLM Runner - memprobe（EdgeSLM_AndroidScale Phase 0-A のために追加）
//
// 1つのプロセスに NPU のモデルを順に開いて**保持**し、メモリの推移を記録する。
// 問い:「1プロセスで何 GB まで NPU のモデルを持てるか」「超えたとき、誰がどこで止めるか」。
//
// 通常の task と入口を分けている理由:
//   nativeRunJob は Session を先に開いてから呼ばれるため、**何も載せていない状態**を測れない。
//   memprobe はモデルの開閉を自分で行い、開く前の値から記録する（nativeMemProbe から呼ぶ）。
//
// 途中で SIGKILL されても記録が残るよう、
//   - 1段ごとの記録は JSON Lines で追記する（out/memprobe_<job_id>_steps.jsonl）
//   - 別スレッドが一定間隔で RSS などを CSV に追記する（out/memprobe_<job_id>_samples.csv）
// どちらも1行ごとに fflush する。プロセスが殺されても、書いた分はカーネル側に残る。

#pragma once

#include "third_party/json.hpp"

namespace edgeslm {

// job の主な項目:
//   models      : 開くモデルの相対パス（base_dir 基準）。この順に開く
//   hold        : true なら開いたセッションを保持し続ける（既定 true）。false なら1本ごとに閉じる
//   run_once    : 開いた直後にゼロ入力で1回推論する（既定 true）。遅延確保を確実に起こすため
//   settle_ms   : 各段のあと待ってから、もう一度測る（既定 1000）
//   sample_ms   : サンプリング間隔（既定 100）
//   stop_on_fail: 開けなかったら以降を試さない（既定 true）
//   hold_ms     : 全部開いたあと、閉じずに待つ時間（既定 0）。別プロセスとの同時使用の試験用
//   prereclaim_mb / prereclaim_step_mb / prereclaim_sleep_ms :
//                 各ロードの前に、この量を少しずつ確保して触り、解放する（既定 0 = しない）。
//                 ファイルキャッシュを先に回収させ、ロード時の lmkd の kill を避けるため
//   ep / runtime: 通常の job と同じ（runtime は RunnerService が注入する）
nlohmann::json run_memprobe(const nlohmann::json& job);

// ロード前に mb だけメモリを step_mb ずつ確保して触り、解放する（ファイルキャッシュを先に回収させる）。
// Session::open も job の `prereclaim_mb` があれば呼ぶ。返り値は前後の MemFree などの記録
nlohmann::json prereclaim(long long mb, long long step_mb, int sleep_ms);

}  // namespace edgeslm
