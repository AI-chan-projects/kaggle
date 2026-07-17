"""
여러 판을 반복 실행해서 승/무/패를 집계하는 벤치마크 스크립트.

실행: python -m code.benchmark --n_games 20
(main.py를 python -m code.main으로 실행하는 것과 같은 방식)
"""

import argparse
import json
from collections import Counter

from .main import ChessGameManager


def run_benchmark(n_games=20, max_plies=400, verbose=False, save_each=False):
    game = ChessGameManager()  # env(=kaggle_environments 인스턴스)는 한 번만 만들고 재사용
    results = []

    for i in range(1, n_games + 1):
        print(f"\n=== Game {i}/{n_games} ===")
        result = game.play(
            max_plies=max_plies,
            verbose=verbose,
            save_json=save_each,
            result_path=f"game_result_{i}.json",
        )
        results.append(result)

    _print_summary(results, n_games)
    return results


def _print_summary(results, n_games):
    counts = Counter(r["result"] for r in results)

    # 진단용: 승/패/무 각각의 평균 ply 수를 따로 계산
    # (턴 수 자체는 벤치마크가 아니지만, 승리 턴수 vs 패배 턴수를 나눠보면
    #  "더 빨리 이기는지" vs "더 쉽게 블런더하는지" 구분하는 데 도움이 됨)
    plies_by_result = {}
    for r in results:
        plies_by_result.setdefault(r["result"], []).append(r["plies"])

    print("\n" + "=" * 40)
    print(f"벤치마크 요약 (총 {n_games}판, self-play: White/Black 모두 같은 엔진)")
    print("=" * 40)

    for key, label in [
        ("white_win", "백 승"),
        ("black_win", "흑 승"),
        ("draw", "무승부"),
        ("unfinished", "미완료(max_plies 초과)"),
    ]:
        c = counts.get(key, 0)
        pct = (c / n_games * 100) if n_games else 0
        if key in plies_by_result:
            avg_plies = sum(plies_by_result[key]) / len(plies_by_result[key])
            avg_str = f", 평균 {avg_plies:.1f}수"
        else:
            avg_str = ""
        print(f"  {label}: {c}판 ({pct:.1f}%){avg_str}")

    # 집계 결과를 파일로도 저장 (다음에 다른 엔진/설정과 비교할 때 근거자료로 사용)
    with open("benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_games": n_games,
                "counts": dict(counts),
                "avg_plies_by_result": {
                    k: sum(v) / len(v) for k, v in plies_by_result.items()
                },
                "raw_results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("\n집계 결과가 benchmark_summary.json으로 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="self-play 여러 판 돌려서 승/무/패 집계")
    parser.add_argument("--n_games", type=int, default=20, help="총 게임 판 수")
    parser.add_argument("--max_plies", type=int, default=400, help="판당 최대 진행 수(안전장치)")
    parser.add_argument("--verbose", action="store_true", help="매 턴 보드를 출력할지 여부 (기본: 끔)")
    parser.add_argument("--save_each", action="store_true", help="판마다 game_result_N.json 저장 여부 (기본: 끔)")
    args = parser.parse_args()

    run_benchmark(
        n_games=args.n_games,
        max_plies=args.max_plies,
        verbose=args.verbose,
        save_each=args.save_each,
    )