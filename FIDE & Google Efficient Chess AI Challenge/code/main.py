import time
import os
import json
import pyspiel
from collections import deque
from integration import stub

class ChessGameManager:
    def __init__(self):
        self.env = stub.Environment.setup()
        self.agent_states = None
        self.recent_fens = deque(maxlen=8)  # 최근 실제 국면 기록 (반복 회피용)
        self.pyspiel_state = None  # 매턴 deserialize하지 않고, 실제 둔 수를 그대로 반영해 동기화 유지
    
    def _display_board(self, board, score):
        print("--- 현재 보드 상태 ---")
        for row in board:
            print(" ".join(row))
        print(f"Material Score: {score}")
        print("----------------------")

    @staticmethod
    def _clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
        return None

    def _get_player_info(self, state):
        idx = state.observation.get('currentPlayer', 0)
        name = "백(White)" if idx == 1 else "흑(Black)"
        return idx, name

    def _save_result(self):
        with open("game_result.json", "w") as f:
            json.dump(self.env.toJSON(), f, indent=4)
        print("결과가 game_result.json으로 저장되었습니다.")

    def _get_result(self, ply):
        """
        env가 실제로 종료됐는지(체크메이트/무승부 등) vs max_plies에 걸려 강제 중단된 건지 구분.
        agent_states[1]=White, agent_states[0]=Black 이므로 각각의 reward로 승/무/패 판정.
        """
        if not self.env.done:
            return {"result": "unfinished", "reason": "max_plies_reached", "plies": ply}

        white_reward = self.agent_states[1].reward
        black_reward = self.agent_states[0].reward

        if white_reward == 1:
            result = "white_win"
        elif black_reward == 1:
            result = "black_win"
        else:
            result = "draw"

        return {
            "result": result,
            "white_reward": white_reward,
            "black_reward": black_reward,
            "plies": ply,
        }

    def play(self, max_plies=20):
        self.env.reset()
        self.agent_states = self.env.step([None, None])
        
        print("게임 시작!")
        ply = 0

        while not self.env.done and ply < max_plies:
            current_idx, player_name = self._get_player_info(self.agent_states[0])
            obs = self.agent_states[current_idx].observation
            fen_string = obs.get("observationString")

            board = stub.FENParser.parse(fen_string)
            board_score = stub.BoardEvaluator.get_material_score(board)

            self._display_board(board, board_score)
            print(f"--- 현재 턴: {player_name} (idx: {current_idx}) ---")
            print(f"--- 현재 진행된 턴 수: {ply} ---")

            legal_actions = obs.get("legalActions", [])
            serialized_state = obs.get("serializedGameAndState")

            if not legal_actions:
                print("더 이상 둘 수 있는 수가 없습니다.")
                break

            # 이번에 둔 수를 나중에 반복 회피용으로 기록
            self.recent_fens.append(fen_string)

            if self.pyspiel_state is None:
                # 게임 시작 후 딱 한 번만 문자열에서 복원
                _game, self.pyspiel_state = pyspiel.deserialize_game_and_state(serialized_state)
            history = self.pyspiel_state.history()

            book_move = stub.OpeningBook.lookup(history)
            if book_move is not None:
                chosen_action = book_move
            else:
                # 오프닝 단계(기물 교환이 적고 이론이 잘 정립된 구간)는 얕은 depth로 충분,
                # 중반 이후로 갈수록 깊게 봄
                depth = 2 if len(history) < 10 else 3
                chosen_action = stub.MinimaxSelector.choose_from_state(
                    self.pyspiel_state, depth=depth, recent_fens=self.recent_fens
                )
            actions = [None, None]
            actions[current_idx] = {"submission": chosen_action}

            self.agent_states = self.env.step(actions)
            self.pyspiel_state.apply_action(chosen_action)  # 문자열 재파싱 없이 로컬 state도 같이 진행
            
            print(f"Player {current_idx} ({player_name})가 액션 '{chosen_action}'를 수행했습니다.")
            ply += 1

        result = self._get_result(ply)
        if result["result"] == "unfinished":
            print(f"게임 종료! (미완료 — max_plies={ply}에서 강제 중단)")
        else:
            label = {"white_win": "백 승", "black_win": "흑 승", "draw": "무승부"}[result["result"]]
            print(f"게임 종료! 결과: {label} (총 {ply}수 진행)")
        self._save_result()
        return result

if __name__ == "__main__":
    game = ChessGameManager()
    game.play(max_plies=400)